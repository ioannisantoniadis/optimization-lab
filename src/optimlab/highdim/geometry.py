"""High-dimensional space is geometrically weird in ways low-dimensional intuition
actively misleads about — no single citation needed here, just two direct
measurements: random directions become near-orthogonal as dimension grows (the
building block behind why gradient noise, random projections, and random
initializations behave the way they do in high dimensions), and a high-dimensional
ball's volume concentrates in a thin shell near its surface rather than spreading
through its interior the way a 2D or 3D ball's does.
"""

from __future__ import annotations

import numpy as np


def pairwise_cosine_similarities(dim: int, n_vectors: int = 200, *, seed: int = 0) -> np.ndarray:
    """Cosine similarity of every distinct pair among `n_vectors` random directions in
    `R^dim`. In low dimensions these spread widely between -1 and 1; as `dim` grows they
    concentrate tightly around 0 — two random directions become *almost certainly* close
    to orthogonal, not because of any special construction, just because there are so
    many more ways to be roughly perpendicular than roughly parallel once there are many
    axes to spread across.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_vectors, dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    gram = vectors @ vectors.T
    i, j = np.triu_indices(n_vectors, k=1)
    return gram[i, j]


def ball_shell_volume_fraction(dim: int, shell_thickness: float) -> float:
    """The fraction of a unit ball's volume lying within `shell_thickness` of its
    surface. A `dim`-ball's volume scales as `r^dim`, so the fraction *inside* radius
    `1 - shell_thickness` is `(1 - shell_thickness)^dim` — shrinking toward zero as
    `dim` grows regardless of how thin the shell is. Concretely: almost the entire
    volume of a high-dimensional ball sits in a thin shell just under its surface, not
    spread through the interior the way a circle's or ordinary sphere's is.
    """
    if not 0.0 <= shell_thickness <= 1.0:
        raise ValueError(f"shell_thickness must be in [0, 1], got {shell_thickness}")
    return 1.0 - (1.0 - shell_thickness) ** dim
