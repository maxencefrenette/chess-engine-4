from __future__ import annotations

import torch

from chess_engine_4.training.optimizers import (
    apply_hyperball_,
    hyperball_radius,
    is_hyperball_parameter,
)


def test_hyperball_parameter_selection_is_semantic() -> None:
    matrix = torch.nn.Parameter(torch.ones(4, 4))
    stacked = torch.nn.Parameter(torch.ones(8, 4, 4))

    assert is_hyperball_parameter("blocks.0.layer.fc1_weight", matrix)
    assert is_hyperball_parameter("blocks.1.layer.fc2_weight", matrix)
    assert is_hyperball_parameter("blocks.0.experts.0.weight0", matrix)
    assert is_hyperball_parameter("blocks.0.experts.gate_up_weight", stacked)
    assert not is_hyperball_parameter("input.weight", matrix)
    assert not is_hyperball_parameter("policy_head.weight", matrix)
    assert not is_hyperball_parameter("blocks.0.router.weight", matrix)
    assert not is_hyperball_parameter("blocks.0.norm.weight", torch.nn.Parameter(torch.ones(4)))


def test_hyperball_update_preserves_matrix_radius() -> None:
    weight = torch.randn(5, 7, dtype=torch.float32)
    update = torch.randn_like(weight)
    radius = hyperball_radius(weight)
    before = weight.clone()

    apply_hyperball_([weight], [update], [radius], lr=0.01, eps=1e-8)

    torch.testing.assert_close(torch.linalg.vector_norm(weight), radius)
    assert not torch.equal(weight, before)


def test_hyperball_update_preserves_each_expert_radius() -> None:
    weight = torch.randn(8, 5, 7, dtype=torch.float32)
    update = torch.randn_like(weight)
    radius = hyperball_radius(weight)

    apply_hyperball_([weight], [update], [radius], lr=0.02, eps=1e-8)

    torch.testing.assert_close(
        torch.linalg.vector_norm(weight, dim=(-2, -1), keepdim=True),
        radius,
    )


def test_hyperball_zero_update_is_noop() -> None:
    weight = torch.randn(5, 7, dtype=torch.float32)
    before = weight.clone()

    apply_hyperball_(
        [weight],
        [torch.zeros_like(weight)],
        [hyperball_radius(weight)],
        lr=0.01,
        eps=1e-8,
    )

    torch.testing.assert_close(weight, before)
