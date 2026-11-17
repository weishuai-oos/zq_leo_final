"""Coordinate transforms shared by T800 boxing command sources."""

from __future__ import annotations

import torch


def rotate_planar_velocity(
    velocity: torch.Tensor,
    *,
    cosine: float,
    sine: float,
) -> torch.Tensor:
    """Rotate the final XY dimension while preserving dtype, device, and leading dimensions."""
    if velocity.shape[-1] != 2:
        raise ValueError(f"Expected planar velocity with final dimension 2, got shape {tuple(velocity.shape)}.")

    rotated = torch.empty_like(velocity)
    rotated[..., 0] = cosine * velocity[..., 0] - sine * velocity[..., 1]
    rotated[..., 1] = sine * velocity[..., 0] + cosine * velocity[..., 1]
    return rotated
