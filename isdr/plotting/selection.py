"""Shared operating-point selection for embedding panels and result tables."""

from __future__ import annotations

import numpy as np


def pareto_knee_index(trust, energy) -> int:
    trust, energy = np.asarray(trust), np.asarray(energy)
    valid = np.isfinite(trust) & np.isfinite(energy)
    front = [
        i
        for i in np.flatnonzero(valid)
        if not np.any(valid & (trust >= trust[i]) & (energy <= energy[i]) & ((trust > trust[i]) | (energy < energy[i])))
    ]
    if not front:
        raise ValueError("No finite operating points")
    points = np.column_stack([trust[front], energy[front]])
    points = (points - points.min(axis=0)) / np.maximum(np.ptp(points, axis=0), 1e-12)
    start = max(range(len(front)), key=lambda i: (trust[front[i]], -energy[front[i]]))
    end = min(range(len(front)), key=lambda i: (energy[front[i]], -trust[front[i]]))
    segment = points[end] - points[start]
    offset = points - points[start]
    distance = np.abs(segment[0] * offset[:, 1] - segment[1] * offset[:, 0])
    # Stable ties prefer greater trust, then lower energy, then the first point.
    best = max(range(len(front)), key=lambda i: (distance[i], trust[front[i]], -energy[front[i]]))
    return int(front[best])
