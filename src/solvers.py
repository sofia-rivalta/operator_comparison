import numpy as np

class Periodic1DAdvectionDiffusion:
    def __init__(self, Pe, r_max, safety_margin):
        self.Pe = Pe
        self.r = safety_margin * r_max if Pe == 0 else min(safety_margin * r_max, 2 / Pe**2) #r is the ratio of dt/dx
        # only set by from_dimensional, kept around for diagnostics
        self.dx = None
        self.alpha = None
        self.dt = None

    @staticmethod #if Pe is not provided you can calculate it in the class
    def peclet(v, dx, alpha):
        #Convert a physical velocity (scalar or array) to a grid Peclet number.
        return v * dx / alpha
    
    @classmethod
    def from_dimensional(cls, dx, alpha, v_max, r_max, safety_margin):
        #Derives Pe from physical quantities; keeps dx/alpha/dt around for diagnostics only.
        eq = cls(Pe=cls.peclet(v_max, dx, alpha), r_max=r_max, safety_margin=safety_margin)
        eq.dx, eq.alpha = dx, alpha
        eq.dt = eq.r * dx**2 / alpha
        return eq

    def step(self, u, Pe = None):
        #Advance state u by one dt.
        if Pe is None:
            Pe = self.Pe
        c = self.r * Pe / 2
        #for u = [a, b ,c , d]
        u_plus = np.roll(u, -1, axis=-1) # u_plus = [b, c, d, a] shift one left
        u_minus = np.roll(u, 1, axis=-1) # u_minus = [d, a, b, c] shift one right
        #note this only works because of the periodic BCs
        return u + self.r * (u_plus - 2 * u + u_minus) - c * (u_plus - u_minus)

class Periodic1DAdvection:
    def __init__(self, CFL, stability_condition, safety_margin):
        self.CFL = CFL
        self.stability_condition = stability_condition
        self.safety_margin = safety_margin
        self.dx = None
        self.v = None
        self.dt = None
    
    @staticmethod
    def compute_CFL(v,dx,dt):
        #compute the CFL numbers from physical values
        return v*dt/dx

    @classmethod
    def from_dimensional(cls, dx, dt, v, stability_condition, safety_margin):
        v_max = np.max(v)
        eq = cls(CFL=cls.compute_CFL(v_max, dx, dt), stability_condition=stability_condition, safety_margin=safety_margin)
        eq.v, eq.dx, eq.dt = v, dx, dt
        return eq

    def step(self, u, CFL=None):
        if CFL is None:
            CFL = self.CFL
        u_plus = np.roll(u, -1, axis=-1) # u_plus = [b, c, d, a] shift one left
        u_minus = np.roll(u, 1, axis=-1) # u_minus = [d, a, b, c] shift one right
        CFL_max = self.stability_condition*self.safety_margin
        assert abs(CFL) < CFL_max
        if CFL > 0:
            return u - CFL*(u-u_minus)
        else:
            return u - CFL*(u_plus-u)
        
class Periodic1DBurgersEquation:
    """u_t + u u_x = nu u_xx, periodic, via Godunov advection + explicit diffusion. Needs nu > 0."""

    def __init__(self, dx, nu, safety_margin):
        if nu <= 0:
            raise ValueError(f"nu must be > 0 (dt is diffusion-limited), got {nu}")
        self.safety_margin = safety_margin
        self.dx = dx
        self.nu = nu
        self.dt = safety_margin*dx**2/(2*nu)  # diffusion-limited; advective CFL checked every step

    def step(self, u):
        dx = self.dx

        u_left = np.roll(u, 1, axis=-1)    # u_left[i] = u[i-1]
        u_right = np.roll(u, -1, axis=-1)  # u_right[i] = u[i+1]

        def riemann_u(uL, uR):
            #resolves u at the interface between uL (left state) and uR (right state)
            S = (uL + uR) / 2                                          #shock speed, Rankine-Hugoniot
            us = np.where(S > 0, uL, uR)                               #shock case
            ur = np.where(uL > 0, uL, np.where(uR < 0, uR, 0))         #rarefaction case
            return np.where(uL > uR, us, ur)                          #compression -> shock, else rarefaction

        u_imh = riemann_u(u_left, u)   # u at the i-1/2 boundary
        u_iph = riemann_u(u, u_right)  # u at the i+1/2 boundary

        dt = self.dt
        cfl = dt * np.max(np.abs(u)) / dx
        assert cfl < 0.5, f"CFL condition violated: dt*max|u|/dx = {cfl:.4f} >= 0.5" #N must be greater than 20 or this will crash. Now dt stays constant.

        advection = 0.5*dt/dx*(u_imh**2 - u_iph**2)                   # Godunov flux, F(u) = u^2/2
        diffusion = self.nu*dt/dx**2*(u_left - 2*u + u_right)  # explicit viscous term, nu*u_xx

        return u + advection + diffusion