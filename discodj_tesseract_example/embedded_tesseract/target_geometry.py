"""Geometry and rasterization helpers for tesseract demo targets."""
from __future__ import annotations
from itertools import product
import numpy as np


def cube_edges(offset: int) -> list[tuple[int, int]]:
    edges = []
    cube = np.array(list(product([-1.0, 1.0], repeat=3)), dtype=np.float32)
    for i, j in product(range(8), repeat=2):
        if i < j and np.count_nonzero(cube[i] != cube[j]) == 1:
            edges.append((offset + i, offset + j))
    return edges


def straight_tesseract_vertices_edges(
    *,
    res: int,
    size_fraction: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    outer = np.array(list(product([-1.0, 1.0], repeat=3)), dtype=np.float32)
    inner = 0.52 * outer + np.array([0.20, -0.16, 0.18], dtype=np.float32)  # slightly shift inner cube
    vertices = np.concatenate([outer, inner], axis=0)
    edges = cube_edges(0) + cube_edges(8) + [(i, i + 8) for i in range(8)]
    vertices -= vertices.mean(axis=0, keepdims=True)
    vertices /= np.max(np.linalg.norm(vertices, axis=1))
    vertices = 0.5 * res + 0.5 * size_fraction * res * vertices
    return vertices.astype(np.float32), edges


def add_segment_max(
    field: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    amplitude: float,
    radius: float,
    support: float = 3.0,
) -> None:
    res = field.shape[0]
    lo = np.maximum(np.floor(np.minimum(p0, p1) - support * radius).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(p0, p1) + support * radius).astype(int) + 1, res)
    if np.any(hi <= lo):
        return

    xs = np.arange(lo[0], hi[0], dtype=np.float32)
    ys = np.arange(lo[1], hi[1], dtype=np.float32)
    zs = np.arange(lo[2], hi[2], dtype=np.float32)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack((x, y, z), axis=-1)

    segment = p1 - p0
    norm2 = float(np.dot(segment, segment))
    if norm2 == 0.0:
        return

    t = np.sum((points - p0) * segment, axis=-1) / norm2
    t = np.clip(t, 0.0, 1.0)
    closest = p0 + t[..., None] * segment
    dist2 = np.sum((points - closest) ** 2, axis=-1)
    contribution = amplitude * np.exp(-0.5 * dist2 / radius**2)
    target = field[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    np.maximum(target, contribution.astype(np.float32), out=target)


def add_node_max(
    field: np.ndarray,
    center: np.ndarray,
    *,
    amplitude: float,
    radius: float,
    support: float = 3.0,
) -> None:
    res = field.shape[0]
    lo = np.maximum(np.floor(center - support * radius).astype(int), 0)
    hi = np.minimum(np.ceil(center + support * radius).astype(int) + 1, res)
    if np.any(hi <= lo):
        return

    xs = np.arange(lo[0], hi[0], dtype=np.float32)
    ys = np.arange(lo[1], hi[1], dtype=np.float32)
    zs = np.arange(lo[2], hi[2], dtype=np.float32)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    dist2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    contribution = amplitude * np.exp(-0.5 * dist2 / radius**2)
    target = field[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    np.maximum(target, contribution.astype(np.float32), out=target)
