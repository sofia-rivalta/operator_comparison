import numpy as np

def GaussianRandomField1D(field_size, correlation_length, smoothness, rng):
    # Periodic zero-mean field, domain length 1.
    # Fixed Matern amplitude per mode, random phase per mode.
    # Normalized so that max|u0| = 1 (U = 1).
    k = np.fft.rfftfreq(field_size, d=1.0 / field_size)  # cycles per domain
    tau = 1.0 / correlation_length
    power = (k**2 + tau**2) ** (-smoothness)
    power[0] = 0.0  # zero mean, mean of u is conserved by Burgers

    phase = np.exp(1j * rng.uniform(0.0, 2 * np.pi, size=k.shape))

    # For even n, irfft keeps only the real part of the Nyquist mode.
    # Force a real coefficient (random sign) so its amplitude is not reduced.
    if field_size % 2 == 0:
        phase[-1] = rng.choice([-1.0, 1.0])

    field = np.fft.irfft(np.sqrt(power) * phase, n=field_size)
    return field / np.abs(field).max()

def GenerateFourierBasis(field_size, n_modes):
    # Real Fourier modes cos(2 pi k x / L), sin(2 pi k x / L) for k = 1..n_modes on the
    # same periodic grid as rfft. k = 0 is skipped (zero mean, as in GaussianRandomField1D).
    # Returns shape (2 * n_modes, field_size): rows alternate cos/sin with increasing k,
    # i.e. [cos 2 pi x, sin 2 pi x, cos 4 pi x, sin 4 pi x, ...].
    assert 2 * n_modes < field_size, "n_modes must stay below Nyquist to be resolved"
    L = 1  # domain length
    k = np.arange(1, n_modes + 1)  # integer wavenumbers
    x = np.arange(field_size) * L / field_size  # periodic grid, x = L is the same point as x = 0
    arg = 2 * np.pi * np.outer(k, x) / L
    basis = np.empty((2 * n_modes, field_size))
    basis[0::2] = np.cos(arg)  # even rows: cos k = 1..n_modes
    basis[1::2] = np.sin(arg)  # odd rows:  sin k = 1..n_modes
    return basis / np.abs(basis).max(axis=1, keepdims=True)  # each mode normalized to amplitude 1

def SuperimposeFourierModes(n_points, rng):
    # Random amplitude per mode (k = 1..n_points//2-1, below Nyquist) of the full basis, summed
    # into one signal sampled at n_points grid points and normalized so max|signal| = 1, as in
    # GaussianRandomField1D.
    basis = GenerateFourierBasis(n_points, n_points // 2 - 1)
    amplitudes = rng.uniform(-1.0, 1.0, size=len(basis))
    field = (amplitudes[:, None] * basis).sum(axis=0)
    return field / np.abs(field).max()
