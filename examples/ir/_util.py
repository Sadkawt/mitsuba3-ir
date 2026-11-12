"""Small shared helpers for the infrared examples (output paths, image export)."""

import os
import numpy as np


def output_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(d, exist_ok=True)
    return d


def project(points, origin, target, up, fov_deg, width, height):
    """Pixel coordinates (x, y) of world points for a Mitsuba perspective
    camera defined by look_at(origin, target, up) and a horizontal ``fov``."""
    origin, target, up = (np.asarray(v, dtype=np.float64) for v in (origin, target, up))
    f = target - origin
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = np.tan(np.deg2rad(fov_deg) / 2)
    out = []
    for p in np.atleast_2d(np.asarray(points, dtype=np.float64)):
        v = p - origin
        z = v @ f
        x = (v @ r) / (z * t)
        y = (v @ u) / (z * t)
        out.append((width / 2 * (1 + x), height / 2 - width / 2 * y))
    return np.array(out)


def save_exr(img, path):
    import mitsuba as mi
    mi.Bitmap(np.ascontiguousarray(np.asarray(img, dtype=np.float32))).write(path)


def save_temperature_png(T_app, path, vmin=None, vmax=None, title=None, cmap='inferno'):
    """Apparent-temperature image as a PNG with a colour bar (matplotlib)."""
    T_app = np.asarray(T_app)
    vmin = float(T_app.min()) if vmin is None else vmin
    vmax = float(T_app.max()) if vmax is None else vmax
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
        im = ax.imshow(T_app, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('apparent temperature [K]')
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except ImportError:
        import mitsuba as mi
        v = np.clip((T_app - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        rgb = (np.repeat(v[..., None], 3, axis=2) * 255).astype(np.uint8)
        mi.Bitmap(rgb).write(path)
    print(f'wrote {path}')


def save_grid_png(images, titles, path, cmap='inferno', clabel='', vmin=None, vmax=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), dpi=110)
    axes = np.atleast_1d(axes)
    for ax, img, t in zip(axes, images, titles):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_title(t)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if clabel:
            cb.set_label(clabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')
