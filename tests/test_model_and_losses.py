from __future__ import annotations

import math

import pytest
import torch

from chess_engine_4.data.leela import (
    HISTORY_PLANE_COUNT,
    INPUT_PLANE_COUNT,
    POLICY_SIZE,
    RULE50_PLANE_INDEX,
)
from chess_engine_4.model import (
    DenseChessNetConfig,
    Moe64A2ChessNetConfig,
    dense_parameter_count,
    model_config_from_dict,
    moe64a2_parameter_count,
)
from chess_engine_4.model.dense import normalize_lc0_planes
from chess_engine_4.model.export import PortableChessNet
from chess_engine_4.model.moe import _route_slots
from chess_engine_4.training.losses import (
    LossWeights,
    lczero_loss,
    moves_left_loss,
    policy_cross_entropy,
    value_cross_entropy,
    wdl_target_from_q_d,
)
from chess_engine_4.training.packed_input import PlaneInputExpander


def test_dense_chess_net_shapes() -> None:
    model = PortableChessNet(DenseChessNetConfig(d_model=64, depth=2, expansion_ratio=2.0))
    output = model(torch.zeros(3, 112, 8, 8))

    assert tuple(output.policy_logits.shape) == (3, 1858)
    assert tuple(output.wdl_logits.shape) == (3, 3)
    assert tuple(output.moves_left.shape) == (3,)


def test_dense_model_discards_history_beyond_configured_length() -> None:
    config = DenseChessNetConfig(
        d_model=64,
        depth=1,
        expansion_ratio=2.0,
        history_length=2,
    )
    model = PortableChessNet(config)
    baseline = torch.zeros(2, INPUT_PLANE_COUNT, 8, 8)
    changed = baseline.clone()
    changed[:, 26:HISTORY_PLANE_COUNT] = torch.randn_like(changed[:, 26:HISTORY_PLANE_COUNT])

    baseline_output = model(baseline)
    changed_output = model(changed)

    torch.testing.assert_close(baseline_output.policy_logits, changed_output.policy_logits)
    torch.testing.assert_close(baseline_output.wdl_logits, changed_output.wdl_logits)
    torch.testing.assert_close(baseline_output.moves_left, changed_output.moves_left)


def test_history_length_reduces_dense_parameter_count() -> None:
    full = dense_parameter_count(d_model=64, depth=2, history_length=8)
    shortened = dense_parameter_count(d_model=64, depth=2, history_length=2)

    assert full - shortened == (8 - 2) * 13 * 64 * 64


def test_packed_input_expansion_matches_dense_input() -> None:
    model = PortableChessNet(DenseChessNetConfig(d_model=64, depth=2, expansion_ratio=2.0))

    _assert_packed_input_expansion_matches_dense_model(model)


def test_dense_model_normalizes_lc0_rule50_plane() -> None:
    planes = torch.zeros(1, INPUT_PLANE_COUNT, 8, 8)
    planes[:, RULE50_PLANE_INDEX] = 50.0
    planes[:, RULE50_PLANE_INDEX + 1] = 7.0

    normalized = normalize_lc0_planes(planes)

    torch.testing.assert_close(
        normalized[:, RULE50_PLANE_INDEX],
        torch.full((1, 8, 8), 50.0 / 99.0),
    )
    torch.testing.assert_close(
        normalized[:, RULE50_PLANE_INDEX + 1],
        torch.full((1, 8, 8), 7.0),
    )


def test_wdl_target_from_q_d() -> None:
    target = wdl_target_from_q_d(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(target, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_policy_cross_entropy_masks_illegal_moves() -> None:
    logits = torch.tensor([[0.0, 10.0, 0.0]])
    target = (torch.tensor([[0, 2, -1]], dtype=torch.int16), torch.tensor([[1.0, 0.0, 0.0]]))

    loss = policy_cross_entropy(logits, target)

    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)


def test_value_cross_entropy_uses_root_row() -> None:
    logits = torch.tensor([[2.0, 0.0, -2.0]])
    values = torch.zeros(1, 6, 3)
    values[0, 4, 0] = 1.0
    values[0, 4, 1] = 0.0

    loss = value_cross_entropy(logits, values)

    expected = -torch.log_softmax(logits, dim=-1)[0, 0]
    torch.testing.assert_close(loss, expected)


def test_moves_left_loss_uses_root_row_root_m() -> None:
    prediction = torch.tensor([30.0])
    values = torch.zeros(1, 6, 3)
    values[0, 4, 2] = 40.0

    loss = moves_left_loss(prediction, values)

    torch.testing.assert_close(loss, torch.tensor(0.125))


def test_lczero_loss_backpropagates() -> None:
    model = PortableChessNet(DenseChessNetConfig(d_model=64, depth=1, expansion_ratio=2.0))
    planes = torch.randn(2, 112, 8, 8)
    policy = _compact_policy(batch_size=2, indices=[0, 1], probs=[0.75, 0.25])
    values = torch.zeros(2, 6, 3)
    values[:, 4, 0] = 1.0
    values[:, 4, 2] = 42.0

    loss = lczero_loss(model(planes), policy, values, weights=LossWeights())
    loss.task.backward()

    assert torch.isfinite(loss.task)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_lczero_loss_adds_router_aux_without_changing_task_loss() -> None:
    model = PortableChessNet(DenseChessNetConfig(d_model=64, depth=1, expansion_ratio=2.0))
    planes = torch.randn(2, 112, 8, 8)
    output = model(planes)
    output = type(output)(
        policy_logits=output.policy_logits,
        wdl_logits=output.wdl_logits,
        moves_left=output.moves_left,
        router_aux_loss=torch.tensor(1.25, requires_grad=True),
        router_dead_experts=torch.tensor(0),
    )
    policy = _compact_policy(batch_size=2, indices=[0], probs=[1.0])
    values = torch.zeros(2, 6, 3)

    loss = lczero_loss(output, policy, values, weights=LossWeights(router_aux=0.01))

    torch.testing.assert_close(loss.total, loss.task + torch.tensor(0.0125))
    torch.testing.assert_close(loss.aux, torch.tensor(0.0125))
    torch.testing.assert_close(loss.router_aux, torch.tensor(1.25))


def test_moe64a2_config_has_fixed_routing_shape() -> None:
    config = model_config_from_dict(
        {
            "kind": "moe64a2",
            "d_model": 64,
            "depth": 2,
            "expansion_ratio": 2.0,
        }
    )

    assert isinstance(config, Moe64A2ChessNetConfig)
    assert config.num_experts == 64
    assert config.num_active_experts == 2
    with pytest.raises(ValueError, match="unknown key"):
        model_config_from_dict({"kind": "moe64a2", "num_experts": 32})


def test_moe64a2_uses_fused_dispatch_for_large_widths() -> None:
    assert Moe64A2ChessNetConfig(d_model=512).cuda_graph_compatible
    assert not Moe64A2ChessNetConfig(d_model=1024).cuda_graph_compatible


def test_moe64a2_parameter_count_counts_all_experts() -> None:
    count = moe64a2_parameter_count(d_model=64, depth=2, expansion_ratio=2.0)
    without_experts = count - 64 * 3 * 64 * 128

    assert count == 2_212_000
    assert without_experts == 639_136


def test_moe64a2_requires_even_depth() -> None:
    with pytest.raises(ValueError, match="positive even"):
        Moe64A2ChessNetConfig(d_model=64, depth=3)
    with pytest.raises(ValueError, match="positive even"):
        moe64a2_parameter_count(d_model=64, depth=3)


def test_moe_route_slots_match_token_order_within_each_expert() -> None:
    flat_experts = torch.tensor([2, 0, 2, 1, 0, 2])
    tokens_per_expert = torch.tensor([2, 1, 3])

    slots = _route_slots(flat_experts, tokens_per_expert)

    torch.testing.assert_close(slots, torch.tensor([0, 0, 1, 0, 1, 2]))


def _assert_packed_input_expansion_matches_dense_model(model: torch.nn.Module) -> None:
    packed_planes = torch.zeros(3, HISTORY_PLANE_COUNT, 8, dtype=torch.uint8)
    packed_planes[:, 0, 0] = 0x80
    plane_scalars = torch.zeros(3, INPUT_PLANE_COUNT - HISTORY_PLANE_COUNT)
    plane_scalars[:, -1] = 1.0

    dense = torch.zeros(3, INPUT_PLANE_COUNT, 8, 8)
    dense[:, 0, 0, 0] = 1.0
    dense[:, -1] = 1.0

    packed_output = model(PlaneInputExpander()(packed_planes, plane_scalars))
    dense_output = model(dense)

    torch.testing.assert_close(packed_output.policy_logits, dense_output.policy_logits)
    torch.testing.assert_close(packed_output.wdl_logits, dense_output.wdl_logits)
    torch.testing.assert_close(packed_output.moves_left, dense_output.moves_left)


def _compact_policy(
    *,
    batch_size: int,
    indices: list[int],
    probs: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_indices = torch.full((batch_size, 218), -1, dtype=torch.int16)
    policy_probs = torch.zeros(batch_size, 218)
    for offset, (index, probability) in enumerate(zip(indices, probs, strict=True)):
        policy_indices[:, offset] = index
        policy_probs[:, offset] = probability
    return policy_indices, policy_probs


@pytest.mark.parametrize("activation", ["geglu", "gelu", "silu", "srelu", "swiglu"])
def test_portable_model_supports_dense_activations(activation: str) -> None:
    model = PortableChessNet(
        DenseChessNetConfig(
            d_model=64,
            depth=1,
            expansion_ratio=2.0,
            activation=activation,
        )
    )

    output = model(torch.zeros(2, INPUT_PLANE_COUNT, 8, 8))

    assert output.policy_logits.shape == (2, POLICY_SIZE)
    assert output.wdl_logits.shape == (2, 3)
    assert output.moves_left.shape == (2,)


def test_dense_config_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="unsupported activation"):
        DenseChessNetConfig(activation="mish")


def test_model_configs_reject_removed_d32_width() -> None:
    with pytest.raises(ValueError, match="at least 64"):
        DenseChessNetConfig(d_model=32)
    with pytest.raises(ValueError, match="at least 64"):
        Moe64A2ChessNetConfig(d_model=32)


def test_dense_config_rejects_invalid_history_length() -> None:
    with pytest.raises(ValueError, match="history_length"):
        DenseChessNetConfig(history_length=0)
