from __future__ import annotations

import torch

from chess_engine_4.training.cli import _muon_parameter_split
from chess_engine_4.training.muon import BatchedMuon


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
        self.norm = torch.nn.LayerNorm(4)
        self.policy_head = torch.nn.Linear(4, 2)


def test_muon_optimizes_only_hidden_matrix_parameters() -> None:
    model = _TinyModel()

    muon_parameters, adamw_parameters = _muon_parameter_split(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}

    assert {names[id(parameter)] for parameter in muon_parameters} == {
        "blocks.0.weight",
        "blocks.1.weight",
    }
    assert {names[id(parameter)] for parameter in adamw_parameters} == {
        "input.weight",
        "input.bias",
        "blocks.0.bias",
        "blocks.1.bias",
        "norm.weight",
        "norm.bias",
        "policy_head.weight",
        "policy_head.bias",
    }


def test_batched_muon_matches_pytorch_muon_step() -> None:
    reference_parameters = [
        torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16)),
    ]
    batched_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in reference_parameters
    ]
    for index, (reference, batched) in enumerate(
        zip(reference_parameters, batched_parameters, strict=True)
    ):
        gradient = torch.randn_like(reference) * (index + 1)
        reference.grad = gradient.clone()
        batched.grad = gradient.clone()

    reference_optimizer = torch.optim.Muon(
        reference_parameters,
        lr=1e-3,
        weight_decay=0.01,
        adjust_lr_fn="match_rms_adamw",
    )
    batched_optimizer = BatchedMuon(
        batched_parameters,
        lr=1e-3,
        weight_decay=0.01,
    )

    reference_optimizer.step()
    batched_optimizer.step()

    for reference, batched in zip(reference_parameters, batched_parameters, strict=True):
        torch.testing.assert_close(batched, reference)
        torch.testing.assert_close(
            batched_optimizer.state[batched]["momentum_buffer"],
            reference_optimizer.state[reference]["momentum_buffer"],
        )
