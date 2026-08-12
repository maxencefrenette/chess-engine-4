"""Training optimizers and Hyperball matrix selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def _hyperball_group_step(
    weight: torch.Tensor,
    model_weight: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    radius: torch.Tensor,
    lr: torch.Tensor,
    sqrt_bias_correction2: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    grad_float = grad.float()
    exp_avg.lerp_(grad_float, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad_float, grad_float, value=1.0 - beta2)
    update = exp_avg / (exp_avg_sq.sqrt() / sqrt_bias_correction2 + eps)
    update_norm = torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True)
    safe_update_norm = update_norm.clamp_min(eps)
    cosine = (weight * update).sum(dim=(-2, -1), keepdim=True) / (
        radius * safe_update_norm
    )
    relative_radius_sq = weight.square().sum(dim=(-2, -1), keepdim=True) / radius.square()
    projection = (
        relative_radius_sq + lr.square() - 2.0 * lr * cosine
    ).clamp_min(eps).sqrt()
    projected = (weight - lr * radius * update / safe_update_norm) / projection
    projected = torch.where(update_norm > eps, projected, weight)
    weight.copy_(projected)
    model_weight.copy_(projected)


_compiled_hyperball_group_step = torch.compile(_hyperball_group_step, fullgraph=True)


def is_hyperball_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    """Return whether a named model parameter is an MLP matrix constrained by AdamH."""

    if not name.startswith("blocks."):
        return False
    if parameter.ndim == 2:
        return (
            name.endswith((".layer.fc1_weight", ".layer.fc2_weight"))
            or (".experts." in name and name.rsplit(".", 1)[-1].startswith("weight"))
        )
    if parameter.ndim == 3:
        return name.endswith((".experts.gate_up_weight", ".experts.down_weight"))
    return False


def hyperball_parameter_partition(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition trainable parameters into Hyperball matrices and ordinary Adam parameters."""

    eligible: list[torch.nn.Parameter] = []
    excluded: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (eligible if is_hyperball_parameter(name, parameter) else excluded).append(parameter)
    if not eligible:
        raise ValueError("AdamH requires at least one eligible MLP matrix.")
    return eligible, excluded


def hyperball_radius(parameter: torch.Tensor) -> torch.Tensor:
    """Compute one FP32 radius per matrix, including per-expert stacked matrices."""

    value = parameter.detach().float()
    if value.ndim == 2:
        return torch.linalg.vector_norm(value)
    if value.ndim == 3:
        return torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)
    raise ValueError("Hyperball parameters must be 2-D matrices or stacked 3-D matrices.")


@torch.no_grad()
def apply_hyperball_(
    weights: list[torch.Tensor],
    updates: list[torch.Tensor],
    radii: list[torch.Tensor],
    *,
    lr: float,
    eps: float,
) -> None:
    """Apply one in-place Hyperball step to FP32 master weights."""

    if not (len(weights) == len(updates) == len(radii)):
        raise ValueError("weights, updates, and radii must have equal lengths.")
    if lr < 0:
        raise ValueError("lr must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    matrix_indices = [index for index, weight in enumerate(weights) if weight.ndim == 2]
    if matrix_indices:
        matrix_weights = [weights[index] for index in matrix_indices]
        matrix_updates = [updates[index] for index in matrix_indices]
        matrix_radii = [radii[index] for index in matrix_indices]
        update_norms = torch._foreach_norm(matrix_updates)
        safe_update_norms = [torch.clamp_min(norm, eps) for norm in update_norms]
        step_scales = [
            lr * radius / update_norm
            for radius, update_norm in zip(matrix_radii, safe_update_norms, strict=True)
        ]
        relative_steps = torch._foreach_mul(matrix_updates, step_scales)
        trial_weights = torch._foreach_sub(matrix_weights, relative_steps)
        trial_norms = torch._foreach_norm(trial_weights)
        safe_trial_norms = [torch.clamp_min(norm, eps) for norm in trial_norms]
        projection_scales = [
            radius / trial_norm
            for radius, trial_norm in zip(matrix_radii, safe_trial_norms, strict=True)
        ]
        projected = torch._foreach_mul(trial_weights, projection_scales)
        torch._foreach_copy_(matrix_weights, projected)

    for index, weight in enumerate(weights):
        if weight.ndim != 3:
            continue
        update = updates[index]
        radius = radii[index]
        update_norm = torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True)
        direction = update / update_norm.clamp_min(eps)
        trial = weight - lr * radius * direction
        trial_norm = torch.linalg.vector_norm(trial, dim=(-2, -1), keepdim=True)
        weight.copy_(radius * trial / trial_norm.clamp_min(eps))


class AdamHyperball(torch.optim.Optimizer):
    """Adam moments wrapped in per-matrix Frobenius Hyperball constraints."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        eligible, excluded = hyperball_parameter_partition(model)
        parameters = [*eligible, *excluded]
        super().__init__(parameters, {"lr": lr})
        from chess_engine_4.model.transformer_engine import te

        fused_adam = te().optimizers.FusedAdam
        self._eligible = eligible
        self._excluded = excluded
        grouped_indices: dict[tuple[int, ...], list[int]] = {}
        for index, parameter in enumerate(eligible):
            grouped_indices.setdefault(tuple(parameter.shape), []).append(index)
        self._group_indices = list(grouped_indices.values())
        self._model_groups = [
            torch.stack([eligible[index].detach() for index in indices])
            for indices in self._group_indices
        ]
        for indices, model_group in zip(self._group_indices, self._model_groups, strict=True):
            for group_index, parameter_index in enumerate(indices):
                eligible[parameter_index].data = model_group[group_index]
        self._master_groups = [
            torch.stack([eligible[index].detach().float() for index in indices])
            for indices in self._group_indices
        ]
        self._radius_groups = [
            torch.linalg.vector_norm(master, dim=(-2, -1), keepdim=True)
            for master in self._master_groups
        ]
        self._masters: list[torch.Tensor] = [torch.empty(0)] * len(eligible)
        self._radii: list[torch.Tensor] = [torch.empty(0)] * len(eligible)
        for indices, masters, radii in zip(
            self._group_indices,
            self._master_groups,
            self._radius_groups,
            strict=True,
        ):
            for group_index, parameter_index in enumerate(indices):
                self._masters[parameter_index] = masters[group_index]
                self._radii[parameter_index] = radii[group_index]
        self._grad_groups = [torch.zeros_like(group) for group in self._model_groups]
        self._grad_views: list[torch.Tensor] = [torch.empty(0)] * len(eligible)
        for indices, grad_group in zip(self._group_indices, self._grad_groups, strict=True):
            for group_index, parameter_index in enumerate(indices):
                self._grad_views[parameter_index] = grad_group[group_index]
        self._moment_groups = [
            (torch.zeros_like(master), torch.zeros_like(master))
            for master in self._master_groups
        ]
        self._eps = eps
        self._betas = betas
        self._step = 0
        self._lr_tensor = torch.tensor(lr, dtype=torch.float32, device=eligible[0].device)
        self._sqrt_bias_correction2 = torch.ones_like(self._lr_tensor)
        self._adam = (
            fused_adam(
                excluded,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=0.0,
                adam_w_mode=False,
                master_weights=True,
                master_weight_dtype=torch.float32,
            )
            if excluded
            else None
        )

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in [*self._eligible, *self._excluded]:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        lr = float(self.param_groups[0]["lr"])
        if self._adam is not None:
            for group in self._adam.param_groups:
                group["lr"] = lr
            self._adam.step()

        eligible_grads = [parameter.grad for parameter in self._eligible]
        if any(grad is None for grad in eligible_grads):
            raise RuntimeError("AdamH eligible matrix is missing its gradient.")
        torch._foreach_copy_(
            self._grad_views,
            [grad for grad in eligible_grads if grad is not None],
        )

        self._step += 1
        beta2 = self._betas[1]
        sqrt_bias_correction2 = math.sqrt(1.0 - beta2**self._step)
        self._lr_tensor.fill_(lr)
        self._sqrt_bias_correction2.fill_(sqrt_bias_correction2)
        for master, model_weight, grad, moments, radius in zip(
            self._master_groups,
            self._model_groups,
            self._grad_groups,
            self._moment_groups,
            self._radius_groups,
            strict=True,
        ):
            _compiled_hyperball_group_step(
                master,
                model_weight,
                grad,
                moments[0],
                moments[1],
                radius,
                self._lr_tensor,
                self._sqrt_bias_correction2,
                self._betas[0],
                self._betas[1],
                self._eps,
            )
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "param_groups": [{"lr": self.param_groups[0]["lr"]}],
            "adam": self._adam.state_dict() if self._adam is not None else None,
            "step": self._step,
            "moment_groups": self._moment_groups,
            "master_groups": self._master_groups,
            "radius_groups": self._radius_groups,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.param_groups[0]["lr"] = state_dict["param_groups"][0]["lr"]
        if self._adam is not None:
            if state_dict["adam"] is None:
                raise ValueError("AdamH checkpoint is missing excluded-parameter Adam state.")
            self._adam.load_state_dict(state_dict["adam"])
        self._step = int(state_dict["step"])
        saved_moments = state_dict["moment_groups"]
        if len(saved_moments) != len(self._moment_groups):
            raise ValueError("AdamH checkpoint moment count does not match the model.")
        for moments, saved in zip(self._moment_groups, saved_moments, strict=True):
            moments[0].copy_(saved[0])
            moments[1].copy_(saved[1])
        saved_masters = state_dict["master_groups"]
        if len(saved_masters) != len(self._master_groups):
            raise ValueError("AdamH checkpoint master count does not match the model.")
        for master, saved in zip(self._master_groups, saved_masters, strict=True):
            master.copy_(saved)
        for model_group, master_group in zip(
            self._model_groups, self._master_groups, strict=True
        ):
            model_group.copy_(master_group)
        saved_radii = state_dict["radius_groups"]
        if len(saved_radii) != len(self._radius_groups):
            raise ValueError("AdamH checkpoint radius count does not match the model.")
        for radius, saved in zip(self._radius_groups, saved_radii, strict=True):
            radius.copy_(saved)

    @torch.no_grad()
    def metrics(self) -> dict[str, float | int]:
        errors: list[torch.Tensor] = []
        for master, radius in zip(self._masters, self._radii, strict=True):
            current = hyperball_radius(master)
            errors.append((current / radius - 1.0).abs().amax())
        max_error = torch.stack(errors).amax().item() if errors else 0.0
        return {
            "optimizer/hyperball_eligible_tensors": len(self._eligible),
            "optimizer/hyperball_excluded_tensors": len(self._excluded),
            "optimizer/hyperball_radius_error_max": max_error,
            "optimizer/hyperball_relative_step": float(self.param_groups[0]["lr"]),
        }


def optimizer_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float | int]:
    """Return optional optimizer diagnostics without coupling the training loop to AdamH."""

    metrics = getattr(optimizer, "metrics", None)
    return metrics() if callable(metrics) else {}
