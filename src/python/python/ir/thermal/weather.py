"""
One parametrised weather model for the thermal solver.

The user gives a date, a location and a handful of meteorological
parameters; the model returns, for any local time, the air temperature,
humidity, wind, solar irradiance components and downwelling longwave
radiation. The diurnal cycle is periodic (the same day repeats), which is
what the solver needs for its spin-up.

* Air temperature: Parton & Logan (1981), Agricultural Meteorology 23,
  205-216 - sinusoidal during the day (lag 1.86 h after noon-centred
  sinus), exponential decay at night (coefficients a = 1.86, b = 2.20,
  c = -0.17).
* Vapour pressure: Magnus formula (WMO No. 8); precipitable water from the
  surface vapour pressure after Leckner (1978): W = 0.493 e_a / T.
* Sky longwave: Idso (1981) full-spectrum clear-sky emissivity, corrected
  for cloud fraction with Crawford & Duchon (1999):
  eps = clf + (1 - clf) eps_clear.
* Solar: Michalsky position, Bird & Hulstrom clear sky, Kasten & Czeplak
  clouds (see ``solar``).
"""

import datetime as _dt
from dataclasses import dataclass, field
import numpy as np

from . import solar as _solar
from ..atmosphere import sky_emissivity_idso, vapour_pressure

SIGMA_SB = 5.670374419e-8


def celsius(t_c):
    """Convert degrees Celsius to kelvin."""
    return np.asarray(t_c, dtype=np.float64) + 273.15


@dataclass
class Weather:
    """
    Parameters
    ----------
    date : 'YYYY-MM-DD'
    latitude, longitude : degrees (north / east positive)
    utc_offset : local standard time minus UTC [h]
    T_min, T_max : daily minimum / maximum air temperature [K]
        (T_min occurs at sunrise, T_max 1.86 h after the sinusoidal peak,
        Parton & Logan 1981)
    relative_humidity : 0-1 (held constant over the day)
    wind_speed : [m/s] at ~2 m (held constant)
    cloud_cover : 0-1 fraction of sky covered
    aod500 : aerosol optical depth at 500 nm (0.05 very clear ... 0.4 hazy)
    pressure_hPa : surface pressure
    ground_albedo : broadband albedo of the surroundings (0.2 typical)
    ozone_cm : total column ozone [atm-cm]
    precipitable_water_cm : overrides the humidity-derived value when given
    T_deep : deep ground temperature [K]; default: mean of T_min and T_max
    """
    date: str
    latitude: float
    longitude: float
    utc_offset: float = 0.0
    T_min: float = 283.15
    T_max: float = 295.15
    relative_humidity: float = 0.5
    wind_speed: float = 2.0
    cloud_cover: float = 0.0
    aod500: float = 0.1
    pressure_hPa: float = 1013.25
    ground_albedo: float = 0.2
    ozone_cm: float = 0.3
    precipitable_water_cm: float = None
    T_deep: float = None
    # Derived-value cache; not an init field, so dataclasses.replace() starts empty,
    # and keyed on the location/date so that mutating the object invalidates it
    _cache: dict = field(default_factory=dict, repr=False, compare=False, init=False)

    def signature(self):
        """Tuple of all parameters (identifies the forcing for caches)."""
        return (self.date, self.latitude, self.longitude, self.utc_offset, self.T_min, self.T_max,
                self.relative_humidity, self.wind_speed, self.cloud_cover, self.aod500,
                self.pressure_hPa, self.ground_albedo, self.ozone_cm, self.precipitable_water_cm,
                self.T_deep)

    # --- derived quantities -------------------------------------------------
    @property
    def doy(self):
        return _solar.day_of_year(self.date)

    @property
    def T_mean(self):
        return 0.5 * (self.T_min + self.T_max)

    def deep_temperature(self):
        return self.T_mean if self.T_deep is None else self.T_deep

    def vapour_pressure(self, t_hours=None):
        """Water vapour pressure [hPa] (constant RH -> follows T_air)."""
        T = self.air_temperature(12.0 if t_hours is None else t_hours)
        return vapour_pressure(T, self.relative_humidity)

    def precipitable_water(self):
        if self.precipitable_water_cm is not None:
            return self.precipitable_water_cm
        e_pa = self.vapour_pressure(12.0) * 100.0
        return float(0.493 * e_pa / self.T_mean)   # Leckner 1978, e in Pa -> cm

    def utc(self, t_hours):
        """Naive UTC datetime for local standard time ``t_hours`` (may exceed 24)."""
        base = _dt.datetime.fromisoformat(self.date)
        return base + _dt.timedelta(hours=float(t_hours) - self.utc_offset)

    # --- sun --------------------------------------------------------------
    def sun(self, t_hours):
        """Solar elevation/azimuth [deg] at local time(s) ``t_hours``."""
        t = np.atleast_1d(np.asarray(t_hours, dtype=np.float64))
        pos = _solar.solar_position(self.latitude, self.longitude, [self.utc(h) for h in t])
        if np.ndim(t_hours) == 0:
            pos = {k: float(np.asarray(v).ravel()[0]) for k, v in pos.items()}
        return pos

    def sunrise_sunset(self):
        """Local times [h] of sunrise and sunset (elevation = -0.833 deg), via a scan."""
        key = (self.date, self.latitude, self.longitude, self.utc_offset)
        if self._cache.get('rise_set_key') != key:
            self._cache.pop('rise_set', None)
            self._cache['rise_set_key'] = key
        if 'rise_set' not in self._cache:
            t = np.linspace(0.0, 24.0, 24 * 60 + 1)
            el = self.sun(t)['elevation'] + 0.833
            up = el > 0
            if up.all():
                rs = (0.0, 24.0)
            elif not up.any():
                rs = (12.0, 12.0)
            else:
                idx_rise = np.where(up & ~np.roll(up, 1))[0]
                idx_set = np.where(~up & np.roll(up, 1))[0]
                rs = (float(t[idx_rise[0]]) if len(idx_rise) else 0.0,
                      float(t[idx_set[0]]) if len(idx_set) else 24.0)
            self._cache['rise_set'] = rs
        return self._cache['rise_set']

    def irradiance(self, t_hours):
        """
        Direct normal, diffuse horizontal and global horizontal irradiance
        [W m^-2] and the sun direction (elevation, azimuth) at local time(s).
        Returns dict(dni, dhi, ghi, elevation, azimuth).
        """
        pos = self.sun(t_hours)
        zenith = 90.0 - np.asarray(pos['elevation'])
        dni, dhi, ghi = _solar.bird_clear_sky(
            zenith, self.doy, self.pressure_hPa, self.aod500, None,
            self.precipitable_water(), self.ozone_cm, self.ground_albedo)
        dni, dhi, ghi = _solar.cloud_correction(dni, dhi, ghi, zenith, self.cloud_cover)
        return dict(dni=dni, dhi=dhi, ghi=ghi, elevation=pos['elevation'], azimuth=pos['azimuth'])

    # --- air --------------------------------------------------------------
    def air_temperature(self, t_hours):
        """Air temperature [K] at local time(s) ``t_hours`` (Parton & Logan 1981)."""
        a, b, c = 1.86, 2.20, -0.17
        rise, set_ = self.sunrise_sunset()
        DL = max(set_ - rise, 0.5)
        NL = 24.0 - DL
        t = np.mod(np.asarray(t_hours, dtype=np.float64), 24.0)
        Tmin, Tmax = self.T_min, self.T_max
        # Day: T = Tmin + (Tmax - Tmin) sin(pi m / (DL + 2a)), m = hours since sunrise
        m = t - rise
        T_day = Tmin + (Tmax - Tmin) * np.sin(np.pi * m / (DL + 2 * a))
        # Temperature at sunset
        T_set = Tmin + (Tmax - Tmin) * np.sin(np.pi * DL / (DL + 2 * a))
        # Night: exponential decay from T_set towards Tmin
        n = np.where(t > set_, t - set_, t + 24.0 - set_)
        # Parton-Logan night equation: T = Tmin - c + (Tset - Tmin + c) exp(-b n / NL)
        T_night = (Tmin - c) + (T_set - Tmin + c) * np.exp(-b * n / NL)
        is_day = (t >= rise) & (t <= set_)
        T = np.where(is_day, T_day, T_night)
        return float(T) if np.ndim(t_hours) == 0 else T

    def sky_emissivity(self, t_hours):
        """Effective full-spectrum sky emissivity (Idso 1981 + Crawford & Duchon 1999 clouds)."""
        T = self.air_temperature(t_hours)
        e_a = vapour_pressure(T, self.relative_humidity)
        eps_clear = sky_emissivity_idso(T, e_a, 'full')
        clf = float(np.clip(self.cloud_cover, 0.0, 1.0))
        eps = clf + (1.0 - clf) * eps_clear
        return float(eps) if np.ndim(t_hours) == 0 else eps

    def sky_longwave(self, t_hours):
        """Downwelling longwave irradiance on a horizontal surface [W m^-2]."""
        T = self.air_temperature(t_hours)
        return self.sky_emissivity(t_hours) * SIGMA_SB * np.asarray(T) ** 4

    def convection_coefficient(self):
        """
        Exterior convective heat transfer coefficient [W m^-2 K^-1],
        h = 5.7 + 3.8 v (McAdams 1954 / Juerges, as used in Duffie & Beckman).
        """
        return 5.7 + 3.8 * float(self.wind_speed)

    def summary(self):
        rise, set_ = self.sunrise_sunset()
        noon = self.irradiance(12.0)
        return (f"Weather({self.date}, lat {self.latitude:.2f}, lon {self.longitude:.2f}, UTC{self.utc_offset:+g}): "
                f"sunrise {rise:.2f} h, sunset {set_:.2f} h, T_air {self.T_min - 273.15:.1f}..{self.T_max - 273.15:.1f} C, "
                f"RH {self.relative_humidity:.0%}, wind {self.wind_speed} m/s, clouds {self.cloud_cover:.0%}, "
                f"AOD500 {self.aod500}, W {self.precipitable_water():.2f} cm; at 12:00 local: sun el {noon['elevation']:.1f} deg, "
                f"DNI {float(noon['dni']):.0f}, DHI {float(noon['dhi']):.0f}, GHI {float(noon['ghi']):.0f} W/m2, "
                f"L_sky {float(self.sky_longwave(12.0)):.0f} W/m2")
