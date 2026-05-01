"""
Gradio app for Hugging Face Spaces.

Users upload PNG/JPG/NPY -> Otsu binarize + resize -> render filtration
animation with Manim. The render is slow (~3-8 min on free CPU), so the UI
sets expectations clearly and shows the binarized preview while the user
waits.

Layout:
  1) Upload box (image or .npy)
  2) Live binarized preview
  3) Settings (size, invert, quality)
  4) Render button
  5) Result video

Files in this folder for HF Spaces:
  - app.py                (this file, must live at repo root)
  - requirements.txt
  - packages.txt
  - README.md             (with Spaces frontmatter)
  - src/persviz/...       (the package)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from persviz.data import load_image_or_npy, synthetic_phantom
from persviz.edt import signed_edt


def make_preview(file_obj, target_size: int, invert: bool):
    if file_obj is None:
        binary = synthetic_phantom(size=target_size, seed=0)
        caption = "Synthetic phantom (no upload). Click Render."
    else:
        try:
            binary = load_image_or_npy(file_obj.name, target_size=target_size, invert=invert)
        except Exception as e:
            return None, f"Error loading file: {e}"
        caption = f"Binarized: {binary.shape}, solid fraction = {binary.mean():.2f}"
    img = np.zeros((*binary.shape, 3), dtype=np.uint8)
    img[binary == 1] = (42, 49, 64)
    img[binary == 0] = (200, 184, 154)
    return img, caption


def render(file_obj, target_size: int, invert: bool, quality: str, seed: int):
    workdir = Path(tempfile.mkdtemp(prefix="persviz_"))
    try:
        if file_obj is None:
            binary = synthetic_phantom(size=target_size, seed=int(seed))
        else:
            binary = load_image_or_npy(file_obj.name, target_size=target_size, invert=invert)
        input_npy = workdir / "input.npy"
        np.save(input_npy, binary)

        scene_file = REPO_ROOT / "src" / "persviz" / "scene.py"
        out_dir = workdir / "out"
        out_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, "-m", "manim",
            f"-q{quality}",
            "--media_dir", str(out_dir),
            "-o", "filtration.mp4",
            str(scene_file),
            "FiltrationScene",
        ]
        env = {**os.environ, "PERSVIZ_INPUT": str(input_npy.resolve())}

        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            return None, f"Render failed after {elapsed:.0f}s.\n\nstderr:\n{proc.stderr[-2500:]}"

        candidates = list(out_dir.glob("**/*.mp4"))
        if not candidates:
            return None, f"No video file produced.\n\nstdout:\n{proc.stdout[-1500:]}"

        final = REPO_ROOT / "static_output" / f"filtration_{int(time.time())}.mp4"
        final.parent.mkdir(exist_ok=True)
        shutil.copy(candidates[0], final)
        return str(final), f"Done in {elapsed:.0f}s. Phantom size {target_size}, quality -{quality}."
    finally:
        pass


THEME = gr.themes.Soft(primary_hue="amber", secondary_hue="slate")

INTRO_MD = """
# persviz — persistent homology of porous media

Upload a 2D image (PNG / JPG) or a binary `.npy`. You'll get an animation
of the **signed-EDT superlevel filtration**: the slice, its distance field,
the 3D heightmap with sweeping plane, and a synchronized persistence diagram.

**Heads up:** rendering takes 3–8 minutes on the free CPU tier. Pick `low`
quality for the fastest result.

The math: Edelsbrunner–Letscher–Zomorodian (2002), Robins–Wood–Sheppard
(2011). Implementation: GUDHI cubical persistence + Manim.
"""

EXAMPLES_MD = """
**Tips for good results**

- The filtration treats **bright pixels as void** and dark pixels as solid
  (matches typical porous-media micrographs). Use *invert* if your image is
  the other way around.
- Smaller phantom size (96) renders fastest; larger (160) gives more H₁
  generators but a longer wait.
- For testing, leave the upload empty — a synthetic Gaussian-blob phantom
  is used.
"""


with gr.Blocks(theme=THEME, title="persviz", analytics_enabled=False) as demo:
    gr.Markdown(INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(
                label="Upload PNG / JPG / .npy (optional)",
                file_types=["image", ".npy"],
            )
            size = gr.Slider(64, 160, value=128, step=16,
                             label="Resize to (NxN). Smaller = faster.")
            invert = gr.Checkbox(label="Invert (treat dark as void)", value=False)
            quality = gr.Radio(
                choices=[("low (480p, fastest)", "l"),
                         ("medium (720p)", "m"),
                         ("high (1080p, slowest)", "h")],
                value="l",
                label="Render quality",
            )
            seed = gr.Slider(0, 100, value=0, step=1,
                             label="Synthetic seed (used when no upload)")
            preview_btn = gr.Button("Preview binarization", variant="secondary")
            render_btn = gr.Button("Render animation", variant="primary")
            gr.Markdown(EXAMPLES_MD)
        with gr.Column(scale=1):
            preview_img = gr.Image(label="Binarized preview", type="numpy")
            preview_caption = gr.Textbox(label="", interactive=False, show_label=False)
            video_out = gr.Video(label="Filtration animation")
            status = gr.Textbox(label="status", interactive=False, lines=4)

    preview_btn.click(make_preview,
                      inputs=[file_in, size, invert],
                      outputs=[preview_img, preview_caption])
    render_btn.click(render,
                     inputs=[file_in, size, invert, quality, seed],
                     outputs=[video_out, status])

if __name__ == "__main__":
    demo.launch(show_api=False)
