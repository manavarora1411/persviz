# Theory

## The signed Euclidean distance transform

For a binary image $I : \mathbb{Z}^2 \to \{0, 1\}$ with $1$ = solid and $0$ = void, define

$$
\phi(x) = \begin{cases}
+ d(x, \text{solid}) & I(x) = 0 \\
- d(x, \text{void}) & I(x) = 1
\end{cases}
$$

where $d$ is the Euclidean distance to the nearest pixel of the opposite phase. The interface $\phi^{-1}(0)$ is the solid–void boundary. Inside void, $\phi$ grows with depth into the pore; inside solid, $\phi$ becomes more negative with depth into the grain.

The transform is computed in $O(n)$ time per phase via the separable lower-envelope algorithm of Felzenszwalb & Huttenlocher (2012), as wrapped by `scipy.ndimage.distance_transform_edt`.

## Superlevel filtration

Define

$$
X_t = \{ x : \phi(x) \ge t \}.
$$

For $t > \max \phi$, $X_t = \emptyset$. For $t < \min \phi$, $X_t$ is the entire image. The family $\{X_t\}_t$ is a filtration: $X_{t_1} \supseteq X_{t_2}$ whenever $t_1 \le t_2$ — wait, the inclusion goes the *other* way for superlevels. We index decreasing $t$; equivalently, sublevel filtration on $-\phi$. Either is computationally equivalent.

## Cubical persistence

A 2D pixel grid carries the structure of a *cubical complex* $K$: each pixel is a 2-cell, each pixel edge a 1-cell, each pixel corner a 0-cell. With the T-construction, top-dimensional cells are assigned filtration values from $\phi$ and lower cells inherit by the rule

$$
\phi(\sigma) = \min \{ \phi(\tau) : \sigma \subseteq \tau, \dim \tau = 2 \}
$$

(for sublevel; reverse for superlevel).

The persistent homology of $\{X_t\}$ is computed by reducing the boundary matrix of $K$ ordered by filtration values. The output is a multiset of pairs $(b_i, d_i)$ with $b_i > d_i$ in superlevel convention. Each pair records a topological feature born at threshold $b_i$ and killed at $d_i$.

This is the Edelsbrunner–Letscher–Zomorodian (2002) algorithm. We use GUDHI's `CubicalComplex` implementation (Maria et al. 2014).

## What the dimensions mean

For 2D images we have $H_0$ and $H_1$:

- $H_0$ classes track *connected components* of $X_t$. A new component appears at a local maximum of $\phi$ (the deepest point of a void pocket). Two components merge at a saddle.
- $H_1$ classes track *loops* in $X_t$. A loop is born when $X_t$ first wraps around a region of $\{\phi < t\}$ — typically when the void closes around a solid grain.

## Reading the diagram

Each persistence pair $(b, d)$ plotted as a point $(b, d)$ in the plane:

- Points far from the diagonal (large $|b - d|$) are **persistent** features — robust to small perturbations of the input. These correspond to real pores and real loops.
- Points near the diagonal are noise, often from EDT discretization. The standard practice in the porous-media literature (Robins et al. 2011) is to threshold persistence at one pixel.

## Birth–death pairing as geometric data

The matrix reduction not only produces the values $(b_i, d_i)$ but pairs each birth with a specific killing simplex. GUDHI exposes this via `cofaces_of_persistence_pairs()`. The animation uses this pairing to draw, at the moment a feature dies, a visual link from the killing cell back to the birth cell — making explicit the geometric meaning that text exposition can only describe.

## References

- Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological persistence and simplification. *Discrete Comput. Geom.* 28, 511–533.
- Robins, V., Wood, P., Sheppard, A. (2011). Theory and algorithms for constructing discrete Morse complexes from grayscale digital images. *IEEE PAMI* 33(8), 1646–1658.
- Felzenszwalb, P., Huttenlocher, D. (2012). Distance transforms of sampled functions. *Theory of Computing* 8(19), 415–428.
- Maria, C., Boissonnat, J.-D., Glisse, M., Yvinec, M. (2014). The GUDHI library: simplicial complexes and persistent homology. ICMS 2014.
- Bauer, U., Kerber, M., Reininghaus, J., Wagner, H. (2014). PHAT — Persistent Homology Algorithms Toolbox.
