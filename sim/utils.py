"""Small generic math helpers with no domain-specific constants."""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def softmax(u: np.ndarray, axis: int = -1) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    u = u - np.max(u, axis=axis, keepdims=True)
    ex = np.exp(u)
    return ex / np.sum(ex, axis=axis, keepdims=True)
