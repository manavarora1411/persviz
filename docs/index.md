# persviz

Animated visualization of the signed-EDT superlevel filtration for 2D porous media slices, with synchronized persistence diagram and barcode.

## Install

```bash
pip install persviz
```

## Run

```bash
persviz render --out demo.mp4
```

## What it shows

Four synchronized panels: a porous slice with its growing superlevel set, the signed EDT field, the persistence diagram (points appear at death events), and the barcode (bars extend in real time).

The pairing on display is the canonical Edelsbrunner–Letscher–Zomorodian pairing: every dot in the diagram corresponds to a unique pair of cells in the cubical complex, one giving birth, one causing death. The animation makes that correspondence visible.

See [Theory](theory.md) for the math, [Gallery](gallery.md) for examples.
