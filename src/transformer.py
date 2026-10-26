import torch  # core tensor library
from torch import nn  # neural network building blocks (Linear, LayerNorm, Module, ...)
import torch.nn.functional as F  # functional ops, e.g. softmax

class RotaryPositionalEmbeddings(nn.Module):
  #Periodic RoPE: pair i is rotated by 2*pi*k_i*m/N at grid point m, with integer wavenumbers
  #k_i = 1..d/2. Every rotation has period N, so the score between points m and n depends only
  #on (m - n) mod N -- the two ends of the grid are neighbours, matching the periodic solver.

  def __init__(self, d: int):

    super().__init__()
    self.d = d
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):

    # rebuilt whenever N changes, since the period is tied to the grid size
    if self.cos_cached is not None and x.shape[-2] == self.cos_cached.shape[0] and self.cos_cached.device == x.device:
      return

    seq_len = x.shape[-2]
    assert self.d // 2 < seq_len / 2, "wavenumbers k >= N/2 alias onto lower ones on an N-point grid"

    k = torch.arange(1, self.d // 2 + 1, device=x.device).float() # integer wavenumbers [1, 2, ... d/2]

    theta = 2 * torch.pi * k / seq_len # THETA_i = 2*pi*k_i/N, so N*THETA_i is a whole number of turns

    seq_idx = torch.arange(seq_len, device=x.device).float() #Position Index -> [0,1,2...seq-1]

    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]


    self.cos_cached = idx_theta2.cos() #Cache [seq, d], broadcasts over [batch, n_head, seq, d]
    self.sin_cached = idx_theta2.sin()

  def forward(self, x: torch.Tensor):
    # x: [batch, n_head, seq, head_dim]
    self._build_cache(x)
    d_2 = self.d // 2
    neg_half_x = torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]
    return x * self.cos_cached + neg_half_x * self.sin_cached # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

class HiddenLayer(nn.Module):
    #Pre-normalized Transformer block: h += self_attention(LayerNorm(h)); h += MLP(LayerNorm(h)).
    #Input/output shape: [batch, sequence_length, n_embd].

    def __init__(self, n_head, n_embd, mlp_dim_ratio):
        super().__init__()  

        #residual stream parameters
        self.n_head = n_head  # number of attention heads
        self.n_embd = n_embd  # width of the residual stream
        assert n_embd % n_head == 0, "head dimension must be a whole number"
        self.head_dim = n_embd // n_head  # head dimension size (residual dimension divided by #heads)
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim), keeps dot-product magnitudes stable before softmax
        assert self.head_dim % 2 == 0, "RoPE rotates pairs of dimensions, so head_dim must be even"
        self.rope = RotaryPositionalEmbeddings(self.head_dim)  # rotates Q and K by their grid position, so scores depend on relative offset
        hidden_dim = mlp_dim_ratio * n_embd #MLP layer dimension

        #two layer norm functions. layer norm forces mean 0 and unit variance, but then allow the
        #network to learn a useful mean and variance using a learned weight and bias
        #which is why we keep each layernorm seperate
        self.norm_attn = nn.LayerNorm(n_embd) #norm before attention
        self.norm_mlp = nn.LayerNorm(n_embd) #norm before MLP

        #learned matrices for the attention layer
        self.Wq = nn.Linear(n_embd, n_embd, bias=False)  
        self.Wk = nn.Linear(n_embd, n_embd, bias=False)  
        self.Wv = nn.Linear(n_embd, n_embd, bias=False)  
        self.Wo = nn.Linear(n_embd, n_embd, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),  # expand n_embd -> hidden_dim
            nn.GELU(),  # nonlinearity between the two linear layers
            nn.Linear(hidden_dim, n_embd),  # project back down hidden_dim -> n_embd
        )

    def forward(self, h):
        batch, seq, _ = h.shape  # pull out dimensions to use in the reshapes below

        #Attention first: 
        z = self.norm_attn(h)  # normalize before computing attention (pre-norm)

        def split_heads(proj):
            return proj(z).reshape(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
            # apply the linear projection, split the embedding dim into (n_head, head_dim),
            # then move the head dim next to batch -> [batch, n_head, seq, head_dim]

        Q, K, V = split_heads(self.Wq), split_heads(self.Wk), split_heads(self.Wv)  # queries/keys/values for every head, computed at once

        Q, K = self.rope(Q), self.rope(K)  # position enters only through Q and K; V stays position-free

        scores = (Q @ K.transpose(-2, -1)) * self.scale  # scaled similarity between every pair of positions -> [batch, n_head, seq, seq]

        att = F.softmax(scores, dim=-1)  # turn similarities into, per query position, a probability distribution over which positions to attend to

        attended = (att @ V).transpose(1, 2).contiguous().reshape(batch, seq, self.n_embd)
        # weighted sum of values according to attention weights, then merge the heads back into one n_embd-wide vector per position

        attended = self.Wo(attended)  # final learned mixing across the (merged) heads

        h = h + attended  # residual connection: add the attention output back onto the input

        #then the pointwise MLP, on the attention-updated h
        h = h + self.mlp(self.norm_mlp(h))  # residual connection: add the feedforward (MLP) output, computed on a freshly normalized copy of h

        return h


class Transformer(nn.Module):

    def __init__(self, n_head, n_embd, n_layer, mlp_dim_ratio):
        super().__init__()
        self.input_projection = nn.Linear(1, n_embd)  # maps the raw u value at each grid point to n_embd; position comes from RoPE inside attention
        self.attn = nn.ModuleList([
            HiddenLayer(n_head=n_head, n_embd=n_embd, mlp_dim_ratio=mlp_dim_ratio)
            for _ in range(n_layer)
        ])  # stack of n_layer identical Transformer blocks, applied one after another
        self.output_norm = nn.LayerNorm(n_embd)  # final normalization before reading out a prediction
        self.output_projection = nn.Linear(n_embd, 1)  # collapses the embedding at each grid point down to a single scalar prediction

        # Start near the identity map, since time stepping begins as output ~= input.
        nn.init.zeros_(self.output_projection.weight)  # zero the output layer's weights ...
        nn.init.zeros_(self.output_projection.bias)  # ... and bias, so at initialization the predicted increment is exactly 0 (output == input)

    def forward(self, x):
        h = self.input_projection(x.unsqueeze(-1))  # [batch, N] -> [batch, N, 1] -> [batch, N, n_embd]
        for layer in self.attn:
            h = layer(h)  # pass through each Transformer block in turn
        increment = self.output_projection(self.output_norm(h)).squeeze(-1) 
        return x + increment
