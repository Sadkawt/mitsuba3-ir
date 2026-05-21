#!/usr/bin/env python3
"""Benchmark render/build time vs number of emitters: stock area emitters vs
the per-shape radiance field. GPU, low spp, warm-timed (compile excluded).

  PYTHONPATH=build/python ./.venv-build/bin/python radiance_field_demo/scaling_benchmark.py
"""
import sys, time, math
import numpy as np

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "cuda_ad_rgb"
SPP     = 16
RES     = 512
COUNTS  = [250, 500, 1000, 2000, 4000, 6000]
FORM_R, LIGHT_R, INTENSITY = 3.0, 0.05, 8.0

import mitsuba as mi
import drjit as dr
mi.set_variant(VARIANT)


def hsv(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def formation(n):
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for k in range(n):
        y = 1.0 - 2.0 * (k + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y * y))
        th = ga * k
        p = [FORM_R * r * math.cos(th), FORM_R * y, FORM_R * r * math.sin(th)]
        rgb = [c * INTENSITY for c in hsv((k / n + 0.25 * (y + 1)) % 1.0, 0.85, 1.0)]
        out.append((p, rgb))
    return out


def scene_dict(mode, lights):
    d = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": {
            "type": "perspective", "fov": 45,
            "to_world": mi.ScalarTransform4f().look_at([0, 0, 9], [0, 0, 0], [0, 1, 0]),
            "film": {"type": "hdrfilm", "width": RES, "height": RES,
                     "rfilter": {"type": "box"}},
            "sampler": {"type": "independent", "sample_count": SPP},
        },
    }
    for i, (p, rgb) in enumerate(lights):
        tw = mi.ScalarTransform4f().translate(p).scale(LIGHT_R)
        if mode == "emitter":
            d[f"l{i}"] = {"type": "sphere", "to_world": tw,
                          "emitter": {"type": "area", "radiance": {"type": "rgb", "value": rgb}}}
        else:
            d[f"l{i}"] = {"type": "sphere", "to_world": tw,
                          "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.0}},
                          "radiance": {"type": "rgb", "value": rgb}}
    return d


def time_one(mode, lights):
    t0 = time.time()
    scene = mi.load_dict(scene_dict(mode, lights))
    dr.sync_thread()
    build = time.time() - t0
    # warm-up (kernel compile), discard
    dr.eval(mi.render(scene, spp=SPP, seed=0)); dr.sync_thread()
    # timed steady-state render
    t1 = time.time()
    dr.eval(mi.render(scene, spp=SPP, seed=1)); dr.sync_thread()
    render = time.time() - t1
    return build, render


def main():
    print(f"variant={VARIANT} spp={SPP} res={RES} counts={COUNTS}")
    res = {"emitter": {"build": [], "render": []},
           "radiance": {"build": [], "render": []}}
    for n in COUNTS:
        lights = formation(n)
        for mode in ("emitter", "radiance"):
            b, r = time_one(mode, lights)
            res[mode]["build"].append(b)
            res[mode]["render"].append(r)
            print(f"  N={n:5d} {mode:8s} build={b:6.3f}s render={r:6.3f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    C = np.array(COUNTS)
    for mode, col, lbl in [("emitter", "tab:red", "default Mitsuba (area emitters, NEE)"),
                           ("radiance", "tab:blue", "radiance field (ours, no NEE)")]:
        ax[0].plot(C, res[mode]["render"], "o-", color=col, label=lbl)
        ax[1].plot(C, res[mode]["build"], "o-", color=col, label=lbl)
    ax[0].set_title(f"Render time vs #emitters  ({VARIANT}, {SPP} spp)")
    ax[1].set_title("Scene construction time vs #emitters")
    for a in ax:
        a.set_xlabel("number of emitters"); a.set_ylabel("seconds")
        a.grid(True, alpha=0.3); a.legend()
    fig.tight_layout()
    out = f"radiance_field_demo/scaling_{VARIANT}.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")
    # speedup summary
    sp = [e / r for e, r in zip(res["emitter"]["render"], res["radiance"]["render"])]
    print("render speedup (emitter/radiance):",
          ", ".join(f"N={n}:{s:.2f}x" for n, s in zip(COUNTS, sp)))


if __name__ == "__main__":
    main()
