from __future__ import annotations

import math

import torch

from chess_engine_4.model import MlpChessNet, MlpChessNetConfig, Transformer64ChessNet
from chess_engine_4.model.heads import AttentionPolicyHead, AttentionPolicyHeadConfig
from chess_engine_4.model.transformer import Transformer64ChessNetConfig
from chess_engine_4.training.losses import (
    LossWeights,
    lczero_loss,
    moves_left_loss,
    policy_cross_entropy,
    value_cross_entropy,
    wdl_target_from_q_d,
)


def test_mlp_chess_net_shapes() -> None:
    model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=2, mlp_ratio=2.0))
    output = model(torch.zeros(3, 112, 8, 8))

    assert tuple(output.policy_logits.shape) == (3, 1858)
    assert tuple(output.wdl_logits.shape) == (3, 3)
    assert tuple(output.moves_left.shape) == (3,)


def test_transformer64_chess_net_shapes() -> None:
    model = Transformer64ChessNet(
        Transformer64ChessNetConfig(
            d_model=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            policy=AttentionPolicyHeadConfig(embedding_size=32, d_model=32),
        )
    )
    output = model(torch.zeros(3, 112, 8, 8))

    assert tuple(output.policy_logits.shape) == (3, 1858)
    assert tuple(output.wdl_logits.shape) == (3, 3)
    assert tuple(output.moves_left.shape) == (3,)


def test_attention_policy_head_uses_lc0_attention_space() -> None:
    head = AttentionPolicyHead(16, config=AttentionPolicyHeadConfig(embedding_size=16, d_model=16))
    tokens = torch.zeros(2, 64, 16)

    logits = head(tokens)

    assert tuple(logits.shape) == (2, 1858)
    assert int(head.policy_map.max()) == 4287


def test_wdl_target_from_q_d() -> None:
    target = wdl_target_from_q_d(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

    torch.testing.assert_close(target, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_policy_cross_entropy_masks_illegal_moves() -> None:
    logits = torch.tensor([[0.0, 10.0, 0.0]])
    target = torch.tensor([[1.0, -1.0, 0.0]])

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
    model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=1, mlp_ratio=2.0))
    planes = torch.randn(2, 112, 8, 8)
    policy = torch.full((2, 1858), -1.0)
    policy[:, 0] = 0.75
    policy[:, 1] = 0.25
    values = torch.zeros(2, 6, 3)
    values[:, 4, 0] = 1.0
    values[:, 4, 2] = 42.0

    loss = lczero_loss(model(planes), policy, values, weights=LossWeights())
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert any(parameter.grad is not None for parameter in model.parameters())
