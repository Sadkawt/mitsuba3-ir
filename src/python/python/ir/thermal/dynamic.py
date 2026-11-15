"""
Linear-response (shading-time) formulation of the surface energy balance.

The column model with the radiative term linearised is a linear
time-invariant system. The surface temperature of a point x at the render
time t is therefore

    T(x, t) = T_ref(t) + sum_k G_per(t - t_k) [q(x, t_k) - q_ref(t_k)] dt,

where G is the surface-temperature response of the column to a unit flux
impulse (one small 1-D solve per material), periodised over the day because
the forcing is periodic, and q(x, t_k) - q_ref is the deviation of the
point's forcing from that of a fully exposed reference element of the same
material. The spatially varying part of q is the direct solar term
alpha_s DNI_k cos(theta_k) V_k(x) (shadow rays) and the sky-view terms
(F_sky(x) - 1) (...), so everything position dependent is a ray-casting
query that the integrator performs at the exact hit point.

This is the response-factor / conduction-transfer-function method of
building thermal simulation (Stephenson & Mitalas 1967; Mitalas &
Stephenson 1967), applied per shading point. See report section 6.8.3.
"""

import numpy as np

from . import solar as _solar
from .heat import Column, SIGMA_SB


def impulse_response(material, h_conv, dt, n_steps, T_lin=293.15):
    """
    Surface-temperature response G(j dt) [K per (W m^-2)] of the column to a
    unit absorbed flux applied during the first step, linearised about a
    uniform state at ``T_lin``.

    The response is obtained from the nonlinear :class:`Column` itself: the
    column is held in equilibrium at ``T_lin`` (air and sky at T_lin, deep
    boundary at T_lin), a flux of 1 W m^-2 is applied during step 0 and the
    surface temperature excursion is recorded. The column's own per-step
    linearisation of the radiative term then supplies 4 eps sigma T_lin^3 on
    every exposed face, and all back-boundary types are handled exactly as in
    the texture solver. Element 0 is the response at the end of the step in
    which the flux is applied.
    """
    m = material.replace(q_internal=0.0)
    if m.back == 'interior' and m.T_internal is not None:
        m = m.replace(T_internal=T_lin)
    col = Column(m, 1, T_lin)
    col.T_deep = T_lin
    L_eq = np.array([SIGMA_SB * T_lin ** 4])
    G = np.zeros(n_steps)
    for j in range(n_steps):
        q = np.array([1.0]) if j == 0 else np.zeros(1)
        col.step(dt, q, L_eq, T_lin, h_conv, T_air_back=T_lin)
        G[j] = col.T[0, 0] - T_lin
    return G


def step_index(t_hours, dt, n_steps):
    """Index of the forcing step whose end lies nearest to local time ``t_hours``
    (the state recorded after that step represents the time ``t_hours``)."""
    return int(round((t_hours % 24.0) * 3600.0 / dt) - 1) % n_steps


def periodise(G, steps_per_day):
    """G_per(j) = sum_n G(j + n * steps_per_day) for one day of lags."""
    n = len(G)
    out = np.zeros(steps_per_day)
    for j in range(steps_per_day):
        out[j] = G[j::steps_per_day].sum() if j < n else 0.0
    return out


class DynamicModel:
    """
    Per-material tables for the ``thermal_dynamic`` emitter.

    Parameters
    ----------
    weather : Weather
    material : Material
    dt : forcing step [s] (default 600)
    days_memory : length of the impulse response [days] (default 6)
    reference_normal : world-space normal of the reference element (the
        orientation about which the response is linearised); default: zenith
    up, north : world frame
    ref_days : spin-up days of the reference (nonlinear) column
    """

    def __init__(self, weather, material, dt=600.0, days_memory=6, reference_normal=None,
                 up=(0, 0, 1), north=(0, 1, 0), ref_days=4):
        self.w = weather
        self.m = material
        self.up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        self.north = np.asarray(north, dtype=np.float64)
        n_steps = int(round(86400.0 / dt))
        self.dt = 86400.0 / n_steps
        self.n_steps = n_steps
        self.t_grid = (np.arange(n_steps) + 0.5) * self.dt / 3600.0
        nref = self.up if reference_normal is None else np.asarray(reference_normal, dtype=np.float64)
        self.n_ref = nref / np.linalg.norm(nref)

        # Forcing on the diurnal grid
        irr = weather.irradiance(self.t_grid)
        self.dni = np.asarray(irr['dni'], dtype=np.float64)
        self.dhi = np.asarray(irr['dhi'], dtype=np.float64)
        self.ghi = np.asarray(irr['ghi'], dtype=np.float64)
        self.T_air = np.asarray(weather.air_temperature(self.t_grid), dtype=np.float64)
        self.L_sky = np.asarray(weather.sky_longwave(self.t_grid), dtype=np.float64)
        self.sun = _solar.sun_direction(irr['elevation'], irr['azimuth'], self.up, self.north)
        self.cos_ref = np.clip(self.sun @ self.n_ref, 0.0, None) * (self.dni > 0)

        # Reference element: full nonlinear column, exposed (V = 1, F_sky = 1)
        h = weather.convection_coefficient()
        col = Column(material, 1, weather.T_mean)
        col.T_deep = weather.deep_temperature()
        self.T_ref = np.zeros(n_steps)
        for day in range(ref_days):
            for s in range(n_steps):
                E_abs = material.alpha_solar * (self.dni[s] * self.cos_ref[s] + self.dhi[s])
                col.step(self.dt, np.array([E_abs]), np.array([self.L_sky[s]]), self.T_air[s], h,
                         T_air_back=self.T_air[s])
                if day == ref_days - 1:
                    self.T_ref[s] = col.T[0, 0]
        T_lin = float(self.T_ref.mean())
        self.h_conv = h
        self.h_total = h + 4.0 * material.emissivity * SIGMA_SB * T_lin ** 3

        # Impulse response (about the mean reference temperature), periodised
        G = impulse_response(material, h, self.dt, int(days_memory) * n_steps, T_lin)
        self.G_per = periodise(G, n_steps)                 # K per (W m^-2) for a lag of j steps

    def _lags(self, t_hours):
        """Lag index (steps) from each forcing step to the render time."""
        s_now = step_index(t_hours, self.dt, self.n_steps)
        k = np.arange(self.n_steps)
        return (s_now - k) % self.n_steps

    def tables(self, t_hours):
        """
        Parameters of the ``thermal_dynamic`` emitter at local time ``t_hours``:
        weights w_k [K] (= G_per(lag_k) DNI_k), sun directions, T_ref, C_ref, C_sky.
        """
        lag = self._lags(t_hours)
        g = self.G_per[lag]
        w = g * self.dni                                   # K per unit cos(theta) * alpha
        # Sky-view terms: temperature change per unit (F_sky - 1)
        q_sky = (self.m.alpha_solar * (self.dhi - self.w.ground_albedo * self.ghi)
                 + self.m.emissivity * (self.L_sky - SIGMA_SB * self.T_air ** 4))
        c_sky = float(np.sum(g * q_sky))
        c_ref = float(np.sum(w * self.cos_ref))
        s_now = step_index(t_hours, self.dt, self.n_steps)
        # Only daylight steps (DNI > 0) are kept. Within one day these form a
        # contiguous run of steps, so consecutive table entries are consecutive
        # forcing steps; the shading-time integrator relies on this for its
        # within-step jitter of the sun direction.
        keep = w != 0.0
        return dict(
            t_ref=float(self.T_ref[s_now]),
            alpha_solar=float(self.m.alpha_solar),
            weights=w[keep],
            sun_directions=self.sun[keep],
            c_ref=c_ref,
            c_sky=c_sky,
            h_total=float(self.h_total),
            h_conv=float(self.h_conv),
            emissivity=float(self.m.emissivity),
        )

    def baselines(self, t_hours):
        """(h_air, s_ref): sum_k G_per(lag_k) sigma T_air,k^4 and sum_k G_per(lag_k) rho_g GHI_k [K]."""
        g = self.G_per[self._lags(t_hours)]
        return float(np.sum(g * SIGMA_SB * self.T_air ** 4)), float(np.sum(g * self.w.ground_albedo * self.ghi))

    def convolve_history(self, t_hours, history):
        """sum_k G_per(lag_k) history[k, :] for a (n_steps, E) history [W m^-2] -> K per unit view factor."""
        g = self.G_per[self._lags(t_hours)]
        return g @ np.asarray(history, dtype=np.float64)

    def emitter_dict(self, t_hours, twosided=True):
        """``thermal_dynamic`` emitter dictionary at local time ``t_hours``."""
        tb = self.tables(t_hours)
        return {
            'type': 'thermal_dynamic',
            't_ref': tb['t_ref'],
            'alpha_solar': tb['alpha_solar'],
            'weights': ', '.join(f'{x:.7g}' for x in tb['weights']),
            'sun_directions': ', '.join(f'{x:.7g}' for x in tb['sun_directions'].ravel()),
            'c_ref': tb['c_ref'],
            'c_sky': tb['c_sky'],
            'h_total': tb['h_total'],
            'h_conv': tb['h_conv'],
            'emissivity': tb['emissivity'],
            'twosided': bool(twosided),
        }

    def temperature(self, t_hours, cos_theta_history, F_sky=1.0):
        """
        Reference implementation of the shading-time formula in NumPy: the
        temperature of a point whose direct-solar visibility history is given
        by ``cos_theta_history`` (array over the forcing steps: cos(theta_k) V_k).
        """
        lag = self._lags(t_hours)
        g = self.G_per[lag]
        w = g * self.dni
        q_sky = (self.m.alpha_solar * (self.dhi - self.w.ground_albedo * self.ghi)
                 + self.m.emissivity * (self.L_sky - SIGMA_SB * self.T_air ** 4))
        dT = self.m.alpha_solar * (np.sum(w * np.asarray(cos_theta_history)) - np.sum(w * self.cos_ref)) \
            + (F_sky - 1.0) * np.sum(g * q_sky)
        s_now = step_index(t_hours, self.dt, self.n_steps)
        T_mid = self.T_ref[s_now] + 0.5 * dT
        h_act = self.h_conv + 4.0 * self.m.emissivity * SIGMA_SB * T_mid ** 3
        corr = self.h_total / max(h_act, 0.25 * self.h_total)
        return self.T_ref[s_now] + dT * corr
