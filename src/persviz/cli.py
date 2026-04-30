"""Command-line entry point: persviz render input.npy --out out.mp4 ..."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persviz")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser("render", help="render filtration animation")
    p_render.add_argument("input", nargs="?", default=None,
                          help="path to 2D binary .npy (omit to use synthetic phantom)")
    p_render.add_argument("--out", "-o", default="filtration.mp4",
                          help="output video path")
    p_render.add_argument("--quality", "-q", default="m",
                          choices=["l", "m", "h", "k"],
                          help="manim quality flag (l/m/h/k)")
    p_render.add_argument("--preview", "-p", action="store_true",
                          help="open video after render")

    p_diag = sub.add_parser("diagram", help="print persistence diagram only")
    p_diag.add_argument("input", help="path to 2D binary .npy")
    p_diag.add_argument("--threshold", "-t", type=float, default=0.0)

    args = parser.parse_args(argv)

    if args.cmd == "render":
        scene_file = Path(__file__).parent / "scene.py"
        cmd = [
            sys.executable, "-m", "manim",
            f"-q{args.quality}",
            "-o", args.out,
            str(scene_file),
            "FiltrationScene",
        ]
        if args.preview:
            cmd.append("-p")
        env = {}
        if args.input:
            env["PERSVIZ_INPUT"] = str(Path(args.input).resolve())
        import os
        return subprocess.call(cmd, env={**os.environ, **env})

    if args.cmd == "diagram":
        import numpy as np
        from persviz.data import load_binary_npy
        from persviz.edt import signed_edt
        from persviz.persistence import superlevel_persistence

        binary = load_binary_npy(args.input)
        phi = signed_edt(binary)
        pairs = superlevel_persistence(phi)
        print(f"{'dim':>4} {'birth':>10} {'death':>10} {'persistence':>12}")
        for p in pairs:
            if np.isfinite(p.death) and p.persistence < args.threshold:
                continue
            d = f"{p.death:.3f}" if np.isfinite(p.death) else "  -inf"
            print(f"{p.dim:>4} {p.birth:>10.3f} {d:>10} {p.persistence:>12.3f}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
