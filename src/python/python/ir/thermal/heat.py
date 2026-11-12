"""
Material properties and the one-dimensional transient conduction solver.

Each surface element owns a column of N nodes (finite volumes) in depth.
The surface node receives the net flux

    q = alpha_s E_abs + eps (L_eff - sigma T_s^4) + h (T_air - T_s) + q_int,surface

and the back node exchanges with a deep ground temperature, is adiabatic,
or is coupled to an interior temperature through a conductance (walls,
vehicle skins), with an optional internal heat flux. The system is advanced
with backward Euler (unconditionally stable); the T^4 term is linearised
about the previous temperature (radiative coefficient 4 eps sigma T^3),
and the tridiagonal system of every column is solved with the Thomas
algorithm, vectorised over all elements.

References: Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and
Mass Transfer (7th ed.); Jacobs, Thermal Infrared Characterization of
Ground Targets and Backgrounds (SPIE, 2nd ed. 2006); Oke, Boundary Layer
Climates (2nd ed. 1987), Table 2.1 for the thermal properties of natural
and urban materials.
"""

from dataclasses import dataclass, replace
import numpy as np

SIGMA_SB = 5.670374419e-8


@dataclass(frozen=True)
class Material:
    """
    Thermal description of a surface element.

    k, rho, c         conductivity [W/m/K], density [kg/m^3], specific heat [J/kg/K]
    alpha_solar       broadband solar absorptance (0-1)
    emissivity        broadband longwave (hemispherical) emissivity (0-1)
    thickness         slab thickness [m]; None = semi-infinite (ground)
    depth             modelled depth for semi-infinite columns [m]
    back              boundary at the back of a slab: 'adiabatic', 'air'
                      (convection + radiation to ambient air, e.g. a free-standing
                      sheet), 'interior' (conductance U_internal to T_internal)
    T_internal        interior temperature [K] for back='interior'
    U_internal        conductance to the interior [W/m^2/K]
    q_internal        internal heat flux [W/m^2] applied at the back node
                      (positive heats the element); e.g. engine heat
    n_nodes           number of nodes in depth
    name              label
    """
    k: float
    rho: float
    c: float
    alpha_solar: float = 0.7
    emissivity: float = 0.9
    thickness: float = None
    depth: float = 1.0
    back: str = 'adiabatic'
    T_internal: float = None
    U_internal: float = 5.0
    q_internal: float = 0.0
    n_nodes: int = 24
    name: str = 'material'

    def replace(self, **kw):
        return replace(self, **kw)

    @property
    def diffusivity(self):
        return self.k / (self.rho * self.c)

    def node_thicknesses(self):
        """Finite-volume thicknesses [m] from the surface downwards."""
        if self.thickness is None:
            # Geometric growth from a thin surface layer to the modelled depth
            n = self.n_nodes
            r = 1.25
            first = self.depth * (r - 1.0) / (r ** n - 1.0)
            dz = first * r ** np.arange(n)
            return dz
        n = max(int(self.n_nodes), 1)
        return np.full(n, self.thickness / n)


#: Typical outdoor materials. Thermal properties after Oke (1987) Table 2.1
#: and Incropera et al.; solar absorptance / emissivity after Jacobs (2006)
#: and Vollmer & Moellmann (2018). Values are representative, not measured
#: for a specific sample; override as needed.
MATERIALS = {
    'asphalt': Material(k=0.75, rho=2110.0, c=920.0, alpha_solar=0.90, emissivity=0.95,
                        thickness=None, depth=1.0, name='asphalt (semi-infinite)'),
    'concrete': Material(k=1.40, rho=2300.0, c=880.0, alpha_solar=0.65, emissivity=0.92,
                         thickness=None, depth=1.0, name='concrete (semi-infinite)'),
    'concrete_wall': Material(k=1.40, rho=2300.0, c=880.0, alpha_solar=0.65, emissivity=0.92,
                              thickness=0.30, back='interior', T_internal=293.15, U_internal=3.0,
                              n_nodes=16, name='concrete wall 0.3 m, interior 20 C'),
    'brick_wall': Material(k=0.70, rho=1900.0, c=840.0, alpha_solar=0.65, emissivity=0.93,
                           thickness=0.25, back='interior', T_internal=293.15, U_internal=3.0,
                           n_nodes=16, name='brick wall 0.25 m, interior 20 C'),
    'soil_dry': Material(k=0.30, rho=1600.0, c=800.0, alpha_solar=0.80, emissivity=0.94,
                         thickness=None, depth=1.0, name='dry sandy soil (semi-infinite)'),
    'soil_wet': Material(k=1.50, rho=2000.0, c=1480.0, alpha_solar=0.85, emissivity=0.96,
                         thickness=None, depth=1.0, name='wet soil (semi-infinite)'),
    'grass': Material(k=0.40, rho=400.0, c=1800.0, alpha_solar=0.75, emissivity=0.97,
                      thickness=None, depth=0.6, name='short grass over soil (approx., no evapotranspiration)'),
    'steel_sheet_painted': Material(k=50.0, rho=7850.0, c=490.0, alpha_solar=0.75, emissivity=0.90,
                                    thickness=0.002, back='air', n_nodes=2,
                                    name='painted steel sheet 2 mm, free-standing'),
    'steel_sheet_insulated': Material(k=50.0, rho=7850.0, c=490.0, alpha_solar=0.75, emissivity=0.90,
                                      thickness=0.002, back='adiabatic', n_nodes=2,
                                      name='painted steel sheet 2 mm, insulated back'),
    'aluminium_sheet_bare': Material(k=237.0, rho=2700.0, c=900.0, alpha_solar=0.20, emissivity=0.05,
                                     thickness=0.002, back='air', n_nodes=2,
                                     name='bare aluminium sheet 2 mm'),
    'wood': Material(k=0.15, rho=600.0, c=1700.0, alpha_solar=0.60, emissivity=0.90,
                     thickness=0.05, back='air', n_nodes=8, name='wood board 5 cm'),
    'water_shallow': Material(k=0.60, rho=1000.0, c=4180.0, alpha_solar=0.94, emissivity=0.98,
                              thickness=0.5, back='adiabatic', n_nodes=10,
                              name='still water 0.5 m (no mixing/evaporation)'),
    'vegetation_canopy': Material(k=0.30, rho=300.0, c=2500.0, alpha_solar=0.70, emissivity=0.97,
                                  thickness=0.01, back='air', n_nodes=2,
                                  name='leaf layer 1 cm, both sides to air (no transpiration)'),
}


class Column:
    """
    Vectorised 1-D conduction columns for E elements sharing one material.

    Parameters
    ----------
    material : Material
    n_elements : int
    T0 : initial temperature [K] (scalar or array of length E)
    """

    def __init__(self, material, n_elements, T0=288.15):
        self.m = material
        self.E = int(n_elements)
        dz = material.node_thicknesses()
        self.N = len(dz)
        self.dz = dz
        self.C = material.rho * material.c * dz                       # J m^-2 K^-1 per node
        self.G = 2.0 * material.k / (dz[:-1] + dz[1:])                # W m^-2 K^-1 between nodes
        self.G_back = 2.0 * material.k / dz[-1]                        # half-node conductance at the back
        self.T = np.full((self.N, self.E), np.nan)
        self.T[:] = np.broadcast_to(np.asarray(T0, dtype=np.float64), (self.E,))[None, :]

    @property
    def surface(self):
        return self.T[0]

    def step(self, dt, E_abs, L_eff, T_air, h, T_air_back=None, L_back=None, q_surface=0.0):
        """
        Advance all columns by ``dt`` seconds.

        E_abs      absorbed solar flux at the surface [W/m^2] (array E)
        L_eff      effective incident longwave [W/m^2] (array E)
        T_air, h   air temperature [K] and convection coefficient [W/m^2/K]
        T_air_back, L_back   ambient temperature / longwave for back='air'
        q_surface  additional surface flux [W/m^2] (array or scalar)
        """
        m = self.m
        N, E = self.N, self.E
        Ts = self.T[0]
        # Linearised radiation: eps sigma T^4 ~ eps sigma Ts^4 + 4 eps sigma Ts^3 (T - Ts)
        h_r = 4.0 * m.emissivity * SIGMA_SB * Ts ** 3
        S0 = E_abs + m.emissivity * (L_eff + 3.0 * SIGMA_SB * Ts ** 4) + h * T_air + q_surface
        H0 = h + h_r

        # Tridiagonal coefficients (per node): a_i T_{i-1} + b_i T_i + c_i T_{i+1} = d_i
        a = np.zeros((N, E))
        b = np.zeros((N, E))
        cc = np.zeros((N, E))
        d = np.zeros((N, E))

        Gup = np.concatenate([[0.0], self.G])       # conductance to the node above
        Gdn = np.concatenate([self.G, [0.0]])       # conductance to the node below
        for i in range(N):
            b[i] = self.C[i] / dt + Gup[i] + Gdn[i]
            a[i] = -Gup[i]
            cc[i] = -Gdn[i]
            d[i] = self.C[i] / dt * self.T[i]
        # Surface boundary
        b[0] += H0
        d[0] += S0
        # Back boundary
        if m.thickness is None or m.back == 'deep':
            T_deep = getattr(self, 'T_deep', None)
            if T_deep is not None:
                b[-1] += self.G_back
                d[-1] += self.G_back * T_deep
        elif m.back == 'air':
            Tb = self.T[-1]
            T_amb = T_air if T_air_back is None else T_air_back
            L_b = (SIGMA_SB * np.asarray(T_amb) ** 4) if L_back is None else L_back
            h_rb = 4.0 * m.emissivity * SIGMA_SB * Tb ** 3
            b[-1] += h + h_rb
            d[-1] += h * T_amb + m.emissivity * (L_b + 3.0 * SIGMA_SB * Tb ** 4)
        elif m.back == 'interior':
            T_int = self.T[-1] if m.T_internal is None else m.T_internal
            b[-1] += m.U_internal
            d[-1] += m.U_internal * T_int
        elif m.back != 'adiabatic':
            raise ValueError(f"unknown back boundary '{m.back}'")
        d[-1] += m.q_internal

        # Thomas algorithm, vectorised over E
        cp = np.zeros((N, E))
        dp = np.zeros((N, E))
        cp[0] = cc[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, N):
            denom = b[i] - a[i] * cp[i - 1]
            cp[i] = cc[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        Tn = np.zeros((N, E))
        Tn[-1] = dp[-1]
        for i in range(N - 2, -1, -1):
            Tn[i] = dp[i] - cp[i] * Tn[i + 1]
        self.T = Tn
        return self.T[0]


def equilibrium_surface_temperature(E_abs, L_eff, T_air, h, emissivity, q=0.0, T0=300.0, iters=50):
    """
    Steady-state surface temperature of an adiabatic element:
    E_abs + eps (L_eff - sigma T^4) + h (T_air - T) + q = 0  (Newton iteration).
    """
    T = np.full_like(np.asarray(E_abs, dtype=np.float64) + 0.0, T0)
    for _ in range(iters):
        f = E_abs + emissivity * (L_eff - SIGMA_SB * T ** 4) + h * (T_air - T) + q
        df = -4.0 * emissivity * SIGMA_SB * T ** 3 - h
        T = T - f / df
    return T
