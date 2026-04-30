---
title: persviz
emoji: 📐
colorFrom: yellow
colorTo: purple
sdk: gradio
python_version: "3.11"
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: mit
short_description: Persistent homology animation for 2D porous media slices
---

# persviz

Animated visualization of the **signed Euclidean distance transform superlevel filtration** for 2D porous-media slices, with synchronized persistence diagram.

The animation reveals the geometric origin of every point in the persistence diagram: each H₀ generator is born when a void component first emerges from the EDT field, and each H₁ generator is born when a region wraps around a solid grain. The pairing displayed matches the boundary-matrix reduction of Edelsbrunner, Letscher & Zomorodian (2002), as implemented in GUDHI's cubical complex.

## Use the live demo

This Space lets you upload a PNG/JPG/NPY and get back a rendered animation. Heads up — renders take 3–8 minutes on the free CPU tier.

## Run locally

```bash
git clone https://github.com/manavarora/persviz
cd persviz
python -m venv .venv && source .venv/bin/activate
pip install -e .
persviz render --out demo.mp4 -q h -p
```

System requirements: ffmpeg, LaTeX. On macOS:

```bash
brew install ffmpeg pango cairo
brew install --cask mactex-no-gui
```

## Pipeline

1. Read 2D input (image or binary `.npy`)
2. Otsu threshold + resize to NxN
3. Compute signed Euclidean distance transform `phi`
4. Run cubical persistence on the superlevel filtration `{phi >= t}` via GUDHI
5. Recover the (birth, death) cell coordinates for each pair
6. Animate t sweeping from `phi.max()` down to `phi.min()` with synchronized panels

## References

- Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological persistence and simplification. *Discrete Comput. Geom.* 28, 511–533.
- Robins, V., Wood, P., Sheppard, A. (2011). Theory and algorithms for constructing discrete Morse complexes from grayscale digital images. *IEEE PAMI* 33(8), 1646–1658.
- Felzenszwalb, P., Huttenlocher, D. (2012). Distance transforms of sampled functions. *Theory of Computing* 8(19), 415–428.
- Maria, C., Boissonnat, J.-D., Glisse, M., Yvinec, M. (2014). The GUDHI library. ICMS 2014.

## License

MIT
