from typing import List

import torch
from torch import nn


class GridEmbeddingND(nn.Module):
    """GridEmbeddingND applies a simple positional embedding as a regular ND grid.
    
    It expects inputs of shape (batch, channels, d_1, ..., d_n)

    Parameters
    ----------
    in_channels : int
        number of channels in input
    dim : int
        dimensions of positional encoding to apply
    grid_boundaries : list, optional
        coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
    def __init__(self, in_channels: int, dim: int=2, grid_boundaries=[[0, 1], [0, 1]]):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(
            grid_boundaries
        ), f"Error: expected grid_boundaries to be an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + self.dim

    def grid(self, spatial_dims: torch.Size, device: str, dtype: torch.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
            Sizes of spatial resolution.
        device : str
            Device to load data (e.g. 'cpu' or 'cuda:*').
        dtype : str
            dtype to encode data.

        Returns
        -------
        torch.tensor
            output grids to concatenate
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(spatial_dims, grid_boundaries=self.grid_boundaries)
            # add batch, channel dims
            grids_by_dim = [x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids_by_dim]
            self._grid = grids_by_dim
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        """
        Parameters
        ----------
        data: torch.Tensor
            assumes shape (batch (optional), channels, x_1, x_2, ...x_n)
        batched: bool
            whether data has a batch dim
        """
        # add batch dim if it doesn't exist
        if not batched:
            if data.ndim == self.dim + 1:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        grids = self.grid(spatial_dims=data.shape[2:], device=data.device, dtype=data.dtype)
        grids = [x.repeat(batch_size, *[1] * (self.dim + 1)) for x in grids]
        out = torch.cat((data, *grids), dim=1)
        return out


def regular_grid_nd(
    resolutions: List[int], grid_boundaries: List[List[int]] = [[0, 1]] * 2
):
    """regular_grid_nd generates a tensor of coordinate points that
    describe a bounded regular grid.

    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid.

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding
    """
    assert len(resolutions) == len(
        grid_boundaries
    ), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start, stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(torch.linspace(start, stop, res + 1)[:-1])
    grid = torch.meshgrid(*meshgrid_inputs, indexing="ij")
    grid = tuple([x.repeat([1] * dim) for x in grid])
    return grid
