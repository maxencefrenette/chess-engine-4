"""Batched Muon optimizer for repeated hidden-layer matrix shapes."""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch

_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


class BatchedMuon(torch.optim.Optimizer):
    """Muon with Newton-Schulz work batched across equal oriented shapes."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(f"BatchedMuon requires 2-D parameters, got {parameter.shape}")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            parameters = [parameter for parameter in group["params"] if parameter.grad is not None]
            if not parameters:
                continue
            gradients = [parameter.grad for parameter in parameters]
            if any(gradient.is_sparse for gradient in gradients):
                raise RuntimeError("BatchedMuon does not support sparse gradients")

            buffers = []
            for parameter, gradient in zip(parameters, gradients, strict=True):
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        gradient,
                        memory_format=torch.preserve_format,
                    )
                buffers.append(state["momentum_buffer"])

            momentum = group["momentum"]
            torch._foreach_lerp_(buffers, gradients, 1 - momentum)
            updates = (
                torch._foreach_lerp(gradients, buffers, momentum)
                if group["nesterov"]
                else buffers
            )
            self._apply_updates(parameters, updates, group)
        return loss

    def _apply_updates(
        self,
        parameters: list[torch.Tensor],
        updates: list[torch.Tensor],
        group: dict[str, Any],
    ) -> None:
        shape_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, update in enumerate(updates):
            shape_groups[_oriented_shape(update)].append(index)

        lr = group["lr"]
        torch._foreach_mul_(parameters, 1 - lr * group["weight_decay"])
        zeropower = _compiled_zeropower() if parameters[0].is_cuda else _batched_zeropower
        for indices in shape_groups.values():
            oriented = [_orient(updates[index]) for index in indices]
            batch = torch.stack(oriented)
            batch = zeropower(
                batch,
                _NS_COEFFICIENTS,
                group["ns_steps"],
                group["eps"],
            )
            adjusted_lr = lr * 0.2 * math.sqrt(max(parameters[indices[0]].shape))
            final_updates = [
                (
                    batch[position].T
                    if updates[index].shape[0] > updates[index].shape[1]
                    else batch[position]
                )
                for position, index in enumerate(indices)
            ]
            torch._foreach_add_(
                [parameters[index] for index in indices],
                final_updates,
                alpha=-adjusted_lr,
            )


def _orient(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.T if tensor.shape[0] > tensor.shape[1] else tensor


def _oriented_shape(tensor: torch.Tensor) -> tuple[int, int]:
    oriented = _orient(tensor)
    return int(oriented.shape[0]), int(oriented.shape[1])


def _batched_zeropower(
    update: torch.Tensor,
    coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    a, b, c = coefficients
    update = update.bfloat16()
    update = update / torch.linalg.vector_norm(
        update,
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(eps)
    for _ in range(ns_steps):
        gram = torch.bmm(update, update.transpose(1, 2))
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.baddbmm(update, gram_update, update, beta=a)
    return update


@functools.cache
def _compiled_zeropower():
    return torch.compile(_batched_zeropower, fullgraph=True)
