"""Evaluate the Skaling law on canonical dense and MoE allocation runs."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

ROOT = Path(__file__).parents[2]
HERE = Path(__file__).parent
FAMILIES = {
    "dense": ROOT / "experiments/best-runs-dense.toml",
    "moe64a2": ROOT / "experiments/best-runs-moe64a2.toml",
}
FIT_RESTARTS = 64
NEW_DENSE_RUNS = HERE / "new-runs.toml"


@dataclass(frozen=True)
class Observation:
    name: str
    width: int
    ratio: float
    total_params: int
    active_params: int
    samples: int
    loss: float
    current_recipe: bool


@dataclass(frozen=True)
class Fit:
    law: str
    params: tuple[float, ...]
    objective: float

    def predict(self, model_size: np.ndarray, data: np.ndarray) -> np.ndarray:
        log_a, log_b, alpha, beta, coupling, floor = self.params
        inner = np.exp(log_a) * model_size**-alpha + np.exp(log_b) * data**-beta
        return inner**coupling + floor


def active_moe_params(total: int, width: int, depth: int = 8) -> int:
    """Count shared parameters plus the two experts active for one token."""
    moe_depth = depth // 2
    hidden = 2 * width
    all_experts = moe_depth * 64 * 3 * width * hidden
    active_experts = moe_depth * 2 * 3 * width * hidden
    return total - all_experts + active_experts


def read_observations(family: str) -> list[Observation]:
    with FAMILIES[family].open("rb") as handle:
        document = tomllib.load(handle)
    observations = []
    for section in ("runs", "allocation_runs"):
        for name, row in document.get(section, {}).items():
            total = int(row["params"])
            width = int(row["d_model"])
            active = active_moe_params(total, width) if family == "moe64a2" else total
            observations.append(
                Observation(
                    name=f"{section}.{name}",
                    width=width,
                    ratio=float(row["training_ratio"]),
                    total_params=total,
                    active_params=active,
                    samples=int(row["samples_seen"]),
                    loss=float(row["loss"]),
                    current_recipe=False,
                )
            )
    if family == "dense":
        with NEW_DENSE_RUNS.open("rb") as handle:
            new_runs = tomllib.load(handle)["runs"]
        for name, row in new_runs.items():
            total = int(row["params"])
            observations.append(
                Observation(
                    name=f"new_runs.{name}",
                    width=int(row["d_model"]),
                    ratio=float(row["training_ratio"]),
                    total_params=total,
                    active_params=total,
                    samples=int(row["samples_seen"]),
                    loss=float(row["loss"]),
                    current_recipe=True,
                )
            )
    else:
        with NEW_DENSE_RUNS.open("rb") as handle:
            new_runs = tomllib.load(handle)["qb_moe_runs"]
        for name, row in new_runs.items():
            if not row.get("fit_eligible", True):
                continue
            total = int(row["params"])
            width = int(row["d_model"])
            observations.append(
                Observation(
                    name=f"new_runs.{name}",
                    width=width,
                    ratio=float(row["training_ratio"]),
                    total_params=total,
                    active_params=active_moe_params(total, width),
                    samples=int(row["samples_seen"]),
                    loss=float(row["loss"]),
                    current_recipe=True,
                )
            )
    return sorted(observations, key=lambda row: (row.width, row.ratio))


def fit_law(
    observations: list[Observation],
    *,
    law: str,
    model_size_field: str,
    restarts: int = FIT_RESTARTS,
) -> Fit:
    model_size = np.array([getattr(row, model_size_field) / 1e6 for row in observations])
    data = np.array([row.samples / 1e8 for row in observations])
    losses = np.array([row.loss for row in observations])
    is_skaling = law == "skaling"
    lower = np.array([-13.8, -13.8, 0.01, 0.01, 0.01, 0.0])
    upper = np.array([16.2, 16.2, 2.0, 2.0, 2.0, 3.0])
    if not is_skaling:
        lower = np.delete(lower, 4)
        upper = np.delete(upper, 4)

    def expand(values: np.ndarray) -> np.ndarray:
        return values if is_skaling else np.insert(values, 4, 1.0)

    def residuals(values: np.ndarray) -> np.ndarray:
        log_a, log_b, alpha, beta, coupling, floor = expand(values)
        inner = np.exp(log_a) * model_size**-alpha + np.exp(log_b) * data**-beta
        predictions = inner**coupling + floor
        return np.log(predictions) - np.log(losses)

    sampler = qmc.Sobol(d=len(lower), scramble=True, seed=20260810)
    starts = qmc.scale(sampler.random_base2(int(math.log2(restarts))), lower, upper)
    best = None
    for initial in starts:
        candidate = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.05,
            max_nfev=20_000,
        )
        objective = float(np.sum(candidate.fun**2))
        if best is None or objective < best.objective:
            best = Fit(
                law=law,
                params=tuple(float(value) for value in expand(candidate.x)),
                objective=objective,
            )
    assert best is not None
    return best


def mape(fit: Fit, observations: list[Observation], model_size_field: str) -> float:
    model_size = np.array([getattr(row, model_size_field) / 1e6 for row in observations])
    data = np.array([row.samples / 1e8 for row in observations])
    observed = np.array([row.loss for row in observations])
    predicted = fit.predict(model_size, data)
    return float(100 * np.mean(np.abs(predicted - observed) / observed))


def fit_shared_skaling_floor(
    moe_model_size_field: str,
    *,
    balance_families: bool = False,
) -> dict[str, object]:
    """Jointly fit dense and d256+ MoE Skaling laws with one shared floor."""
    dense = [
        row
        for row in read_observations("dense")
        if row.current_recipe and row.width >= 64 and row.ratio >= 0.1
    ]
    moe = [row for row in read_observations("moe64a2") if row.width >= 256]

    def arrays(
        observations: list[Observation], model_size_field: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(
                [getattr(row, model_size_field) / 1e6 for row in observations]
            ),
            np.array([row.samples / 1e8 for row in observations]),
            np.array([row.loss for row in observations]),
        )

    dense_n, dense_d, dense_loss = arrays(dense, "total_params")
    moe_n, moe_d, moe_loss = arrays(moe, moe_model_size_field)
    family_lower = np.array([-13.8, -13.8, 0.01, 0.01, 0.01])
    family_upper = np.array([16.2, 16.2, 2.0, 2.0, 2.0])
    lower = np.concatenate((family_lower, family_lower, np.array([0.0])))
    upper = np.concatenate((family_upper, family_upper, np.array([3.0])))

    def predict(
        values: np.ndarray, model_size: np.ndarray, data: np.ndarray, offset: int
    ) -> np.ndarray:
        log_a, log_b, alpha, beta, coupling = values[offset : offset + 5]
        inner = np.exp(log_a) * model_size**-alpha + np.exp(log_b) * data**-beta
        return inner**coupling + values[-1]

    def residuals(values: np.ndarray) -> np.ndarray:
        dense_residuals = np.log(predict(values, dense_n, dense_d, 0)) - np.log(
            dense_loss
        )
        moe_residuals = np.log(predict(values, moe_n, moe_d, 5)) - np.log(moe_loss)
        if balance_families:
            dense_residuals /= math.sqrt(len(dense_residuals))
            moe_residuals /= math.sqrt(len(moe_residuals))
        return np.concatenate((dense_residuals, moe_residuals))

    sampler = qmc.Sobol(d=len(lower), scramble=True, seed=20260811)
    starts = qmc.scale(
        sampler.random_base2(int(math.log2(FIT_RESTARTS))), lower, upper
    )
    best = min(
        (
            least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                loss="huber",
                f_scale=0.05,
                max_nfev=20_000,
            )
            for initial in starts
        ),
        key=lambda candidate: float(np.sum(candidate.fun**2)),
    )

    def family_output(
        observations: list[Observation],
        losses: np.ndarray,
        model_size: np.ndarray,
        data: np.ndarray,
        offset: int,
    ) -> dict[str, object]:
        log_a, log_b, alpha, beta, coupling = best.x[offset : offset + 5]
        predictions = predict(best.x, model_size, data, offset)
        return {
            "observation_count": len(observations),
            "mape": float(100 * np.mean(np.abs(predictions - losses) / losses)),
            "parameters": {
                "A": math.exp(log_a),
                "B": math.exp(log_b),
                "alpha": float(alpha),
                "beta": float(beta),
                "k": float(coupling),
                "E": float(best.x[-1]),
            },
        }

    return {
        "family": "dense+moe64a2",
        "comparison": "shared_skaling_floor",
        "weighting": "family_balanced" if balance_families else "observation_weighted",
        "moe_model_size": moe_model_size_field,
        "dense_minimum_width": 64,
        "dense_minimum_ratio": 0.1,
        "moe_minimum_width": 256,
        "shared_E": float(best.x[-1]),
        "dense": family_output(dense, dense_loss, dense_n, dense_d, 0),
        "moe64a2": family_output(moe, moe_loss, moe_n, moe_d, 5),
    }


def fit_moe_with_dense_data_term(model_size_field: str) -> dict[str, object]:
    """Fit MoE A/alpha/k while reusing dense B/beta/E."""
    dense = [
        row
        for row in read_observations("dense")
        if row.current_recipe and row.width >= 64 and row.ratio >= 0.1
    ]
    dense_fit = fit_law(dense, law="skaling", model_size_field="total_params")
    _, log_b, _, beta, _, floor = dense_fit.params
    moe = [row for row in read_observations("moe64a2") if row.width >= 256]

    def constrained_fit(
        observations: list[Observation], *, restarts: int = FIT_RESTARTS
    ) -> Fit:
        model_size = np.array(
            [getattr(row, model_size_field) / 1e6 for row in observations]
        )
        data = np.array([row.samples / 1e8 for row in observations])
        losses = np.array([row.loss for row in observations])
        lower = np.array([-13.8, 0.01, 0.01])
        upper = np.array([16.2, 2.0, 2.0])

        def expand(values: np.ndarray) -> np.ndarray:
            log_a, alpha, coupling = values
            return np.array([log_a, log_b, alpha, beta, coupling, floor])

        def residuals(values: np.ndarray) -> np.ndarray:
            log_a, _, alpha, _, coupling, _ = expand(values)
            inner = np.exp(log_a) * model_size**-alpha + np.exp(log_b) * data**-beta
            return np.log(inner**coupling + floor) - np.log(losses)

        sampler = qmc.Sobol(d=3, scramble=True, seed=20260811)
        starts = qmc.scale(
            sampler.random_base2(int(math.log2(restarts))), lower, upper
        )
        best = None
        for initial in starts:
            candidate = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                loss="huber",
                f_scale=0.05,
                max_nfev=20_000,
            )
            objective = float(np.sum(candidate.fun**2))
            if best is None or objective < best.objective:
                best = Fit(
                    law="skaling_dense_data_term",
                    params=tuple(float(value) for value in expand(candidate.x)),
                    objective=objective,
                )
        assert best is not None
        return best

    fit = constrained_fit(moe)
    leave_one_out = [
        mape(
            constrained_fit(
                [row for row in moe if row != heldout], restarts=16
            ),
            [heldout],
            model_size_field,
        )
        for heldout in moe
    ]
    max_width = max(row.width for row in moe)
    size_test = [row for row in moe if row.width == max_width]
    size_train = [row for row in moe if row.width < max_width]
    longest = [
        max((row for row in moe if row.width == width), key=lambda row: row.ratio)
        for width in sorted({row.width for row in moe})
    ]
    data_train = [row for row in moe if row not in longest]
    return {
        "family": "moe64a2",
        "comparison": "reuse_dense_B_beta_E",
        "model_size": model_size_field,
        "minimum_width": 256,
        "observation_count": len(moe),
        "parameters": {
            "A": math.exp(fit.params[0]),
            "B": math.exp(fit.params[1]),
            "alpha": fit.params[2],
            "beta": fit.params[3],
            "k": fit.params[4],
            "E": fit.params[5],
        },
        "full_mape": mape(fit, moe, model_size_field),
        "leave_one_out_mape": float(np.mean(leave_one_out)),
        "size_extrapolation_mape": mape(
            constrained_fit(size_train), size_test, model_size_field
        ),
        "data_extrapolation_mape": mape(
            constrained_fit(data_train), longest, model_size_field
        ),
    }


def fit_moe_with_dense_floor(model_size_field: str) -> dict[str, object]:
    """Fit the MoE Skaling law while reusing only dense E."""
    dense = [
        row
        for row in read_observations("dense")
        if row.current_recipe and row.width >= 64 and row.ratio >= 0.1
    ]
    dense_fit = fit_law(dense, law="skaling", model_size_field="total_params")
    floor = dense_fit.params[-1]
    moe = [row for row in read_observations("moe64a2") if row.width >= 256]

    def constrained_fit(
        observations: list[Observation], *, restarts: int = FIT_RESTARTS
    ) -> Fit:
        model_size = np.array(
            [getattr(row, model_size_field) / 1e6 for row in observations]
        )
        data = np.array([row.samples / 1e8 for row in observations])
        losses = np.array([row.loss for row in observations])
        lower = np.array([-13.8, -13.8, 0.01, 0.01, 0.01])
        upper = np.array([16.2, 16.2, 2.0, 2.0, 2.0])

        def expand(values: np.ndarray) -> np.ndarray:
            log_a, log_b, alpha, beta, coupling = values
            return np.array([log_a, log_b, alpha, beta, coupling, floor])

        def residuals(values: np.ndarray) -> np.ndarray:
            log_a, log_b, alpha, beta, coupling, _ = expand(values)
            inner = np.exp(log_a) * model_size**-alpha + np.exp(log_b) * data**-beta
            return np.log(inner**coupling + floor) - np.log(losses)

        sampler = qmc.Sobol(d=5, scramble=True, seed=20260811)
        starts = qmc.scale(
            sampler.random_base2(int(math.log2(restarts))), lower, upper
        )
        best = None
        for initial in starts:
            candidate = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                loss="huber",
                f_scale=0.05,
                max_nfev=20_000,
            )
            objective = float(np.sum(candidate.fun**2))
            if best is None or objective < best.objective:
                best = Fit(
                    law="skaling_dense_floor",
                    params=tuple(float(value) for value in expand(candidate.x)),
                    objective=objective,
                )
        assert best is not None
        return best

    fit = constrained_fit(moe)
    max_width = max(row.width for row in moe)
    size_test = [row for row in moe if row.width == max_width]
    size_train = [row for row in moe if row.width < max_width]
    longest = [
        max((row for row in moe if row.width == width), key=lambda row: row.ratio)
        for width in sorted({row.width for row in moe})
    ]
    data_train = [row for row in moe if row not in longest]
    sparse = l_shape_rows(moe, anchor_width=None, anchor_ratio=0.01)
    sparse_holdout = [row for row in moe if row not in sparse]
    return {
        "family": "moe64a2",
        "comparison": "reuse_dense_E",
        "model_size": model_size_field,
        "minimum_width": 256,
        "observation_count": len(moe),
        "parameters": {
            "A": math.exp(fit.params[0]),
            "B": math.exp(fit.params[1]),
            "alpha": fit.params[2],
            "beta": fit.params[3],
            "k": fit.params[4],
            "E": fit.params[5],
        },
        "full_mape": mape(fit, moe, model_size_field),
        "l_shape_heldout_mape": mape(
            constrained_fit(sparse), sparse_holdout, model_size_field
        ),
        "size_extrapolation_mape": mape(
            constrained_fit(size_train), size_test, model_size_field
        ),
        "data_extrapolation_mape": mape(
            constrained_fit(data_train), longest, model_size_field
        ),
    }


def evaluation_folds(
    observations: list[Observation],
) -> dict[str, list[tuple[list[Observation], list[Observation]]]]:
    widths = sorted({row.width for row in observations})
    ratios = sorted({row.ratio for row in observations})
    common_widths = [
        width
        for width in widths
        if len({row.ratio for row in observations if row.width == width}) >= 2
    ]
    interpolation = [
        row
        for row in observations
        if min(widths) < row.width < max(widths)
        and min(ratios) < row.ratio < max(ratios)
    ]
    max_width = max(widths)
    extrap_n = [row for row in observations if row.width == max_width]
    extrap_d = [
        max((row for row in observations if row.width == width), key=lambda row: row.ratio)
        for width in common_widths
    ]
    far_width = common_widths[-1]
    far_ratio = max(row.ratio for row in observations if row.width == far_width)
    far = [
        row
        for row in observations
        if row.width == far_width and row.ratio == far_ratio
    ]
    return {
        "interpolation": [
            ([candidate for candidate in observations if candidate != row], [row])
            for row in interpolation
        ],
        "extrapolation_n": [
            ([row for row in observations if row.width < max_width], extrap_n)
        ],
        "extrapolation_d": [
            ([row for row in observations if row not in extrap_d], extrap_d)
        ],
        "far_both": [
            (
                [
                    row
                    for row in observations
                    if row.width < far_width and row.ratio < far_ratio
                ],
                far,
            )
        ],
    }


def l_shape_rows(
    observations: list[Observation],
    *,
    anchor_width: int | None = None,
    anchor_ratio: float | None = None,
) -> list[Observation]:
    min_width = min(row.width for row in observations) if anchor_width is None else anchor_width
    # Default to the cheapest eligible corner; callers may select an explicit
    # low-compute size arm when that corner run is diagnostically invalid.
    min_ratio = (
        min(row.ratio for row in observations if row.width == min_width)
        if anchor_ratio is None
        else anchor_ratio
    )
    return [
        row for row in observations if row.width == min_width or row.ratio == min_ratio
    ]


def compare_l_shape_anchors(
    anchor_widths: tuple[int, ...],
    *,
    model_size_field: str,
) -> dict[str, object]:
    observations = [row for row in read_observations("dense") if row.current_recipe]
    training_sets = {
        width: l_shape_rows(observations, anchor_width=width) for width in anchor_widths
    }
    training_union = {
        row for training_rows in training_sets.values() for row in training_rows
    }
    common_holdout = [row for row in observations if row not in training_union]
    anchor_output: dict[str, object] = {}
    errors_by_law: dict[str, dict[int, np.ndarray]] = {
        law: {} for law in ("chinchilla", "skaling")
    }
    for width, training_rows in training_sets.items():
        law_output = {}
        for law in ("chinchilla", "skaling"):
            fit = fit_law(
                training_rows,
                law=law,
                model_size_field=model_size_field,
            )
            model_size = np.array(
                [getattr(row, model_size_field) / 1e6 for row in common_holdout]
            )
            data = np.array([row.samples / 1e8 for row in common_holdout])
            observed = np.array([row.loss for row in common_holdout])
            errors = 100 * np.abs(fit.predict(model_size, data) - observed) / observed
            errors_by_law[law][width] = errors
            law_output[law] = {
                "common_holdout_mape": float(np.mean(errors)),
                "absolute_percentage_errors": {
                    row.name: float(error)
                    for row, error in zip(common_holdout, errors, strict=True)
                },
            }
        anchor_output[str(width)] = {
            "train": [row.name for row in training_rows],
            "laws": law_output,
        }

    rng = np.random.default_rng(20260810)
    bootstrap_indices = rng.integers(
        0, len(common_holdout), size=(20_000, len(common_holdout))
    )
    first, second = anchor_widths
    paired_bootstrap = {}
    for law in ("chinchilla", "skaling"):
        differences = (
            errors_by_law[law][first][bootstrap_indices].mean(axis=1)
            - errors_by_law[law][second][bootstrap_indices].mean(axis=1)
        )
        paired_bootstrap[law] = {
            "mape_difference_first_minus_second": float(np.mean(differences)),
            "percentile_95_interval": [
                float(value) for value in np.percentile(differences, [2.5, 97.5])
            ],
            "probability_first_is_worse": float(np.mean(differences > 0)),
        }

    return {
        "family": "dense",
        "comparison": "l_shape_anchors",
        "model_size": model_size_field,
        "common_holdout": [row.name for row in common_holdout],
        "anchors": anchor_output,
        "paired_cell_bootstrap": paired_bootstrap,
    }


def compare_d32_extrapolation(model_size_field: str) -> dict[str, object]:
    observations = [row for row in read_observations("dense") if row.current_recipe]
    output: dict[str, object] = {}
    for minimum_ratio in (None, 0.1):
        expanded = [
            row
            for row in observations
            if minimum_ratio is None or row.ratio >= minimum_ratio
        ]
        base = [row for row in expanded if row.width >= 64]
        max_width = max(row.width for row in base)
        n_test = [row for row in base if row.width == max_width]
        widths = [
            width
            for width in sorted({row.width for row in base})
            if len([row for row in base if row.width == width]) >= 2
        ]
        d_test = [
            max((row for row in base if row.width == width), key=lambda row: row.ratio)
            for width in widths
        ]
        regime_output = {}
        for regime, test in (("model_size", n_test), ("data_horizon", d_test)):
            floor_output = {}
            for floor, pool in (("d64", base), ("d32", expanded)):
                if regime == "model_size":
                    train = [row for row in pool if row.width < max_width]
                else:
                    train = [row for row in pool if row not in test]
                floor_output[floor] = {
                    law: mape(
                        fit_law(
                            train,
                            law=law,
                            model_size_field=model_size_field,
                        ),
                        test,
                        model_size_field,
                    )
                    for law in ("chinchilla", "skaling")
                }
            regime_output[regime] = {
                "test": [row.name for row in test],
                "anchor_floor_mape": floor_output,
            }
        output["all_ratios" if minimum_ratio is None else "minimum_ratio_0.1"] = (
            regime_output
        )
    return {
        "family": "dense",
        "comparison": "d32_extrapolation_influence",
        "model_size": model_size_field,
        "regimes": output,
    }


def evaluate(
    family: str,
    model_size_field: str,
    *,
    minimum_width: int | None = None,
    minimum_ratio: float | None = None,
    l_shape_width: int | None = None,
    l_shape_ratio: float | None = None,
    current_only: bool = False,
    include_influence: bool = False,
) -> dict[str, object]:
    observations = read_observations(family)
    if current_only:
        observations = [row for row in observations if row.current_recipe]
    if minimum_width is not None:
        observations = [row for row in observations if row.width >= minimum_width]
    if minimum_ratio is not None:
        observations = [row for row in observations if row.ratio >= minimum_ratio]
    laws = ("chinchilla", "skaling")
    full_fits = {
        law: fit_law(observations, law=law, model_size_field=model_size_field) for law in laws
    }
    output: dict[str, object] = {
        "family": family,
        "model_size": model_size_field,
        "minimum_width": minimum_width,
        "minimum_ratio": minimum_ratio,
        "l_shape_ratio": l_shape_ratio,
        "current_only": current_only,
        "observation_count": len(observations),
        "observations": [asdict(row) for row in observations],
        "full": {
            law: {
                "mape": mape(fit, observations, model_size_field),
                "objective": fit.objective,
                "parameters": {
                    "A": math.exp(fit.params[0]),
                    "B": math.exp(fit.params[1]),
                    "alpha": fit.params[2],
                    "beta": fit.params[3],
                    "k": fit.params[4],
                    "E": fit.params[5],
                },
                "residuals": [
                    {
                        "name": row.name,
                        "observed": row.loss,
                        "predicted": float(
                            fit.predict(
                                np.array([getattr(row, model_size_field) / 1e6]),
                                np.array([row.samples / 1e8]),
                            )[0]
                        ),
                    }
                    for row in observations
                ],
            }
            for law, fit in full_fits.items()
        },
        "cross_validation": {},
    }
    for regime, folds in evaluation_folds(observations).items():
        regime_output: dict[str, object] = {
            "fold_count": len(folds),
            "test": [[row.name for row in test] for _, test in folds],
            "laws": {},
        }
        for law in laws:
            parameter_count = 6 if law == "skaling" else 5
            if not folds or min(len(train) for train, _ in folds) <= parameter_count:
                regime_output["laws"][law] = {
                    "available": False,
                    "reason": "insufficient training rows for the fitted parameter count",
                }
                continue
            errors = [
                mape(
                    fit_law(train, law=law, model_size_field=model_size_field),
                    test,
                    model_size_field,
                )
                for train, test in folds
            ]
            regime_output["laws"][law] = {
                "available": True,
                "mape_mean": float(np.mean(errors)),
                "mape_std": float(np.std(errors)) if len(errors) > 1 else None,
            }
        output["cross_validation"][regime] = regime_output

    sparse = l_shape_rows(
        observations,
        anchor_width=l_shape_width,
        anchor_ratio=l_shape_ratio,
    )
    sparse_holdout = [row for row in observations if row not in sparse]
    output["l_shape"] = {
        "train": [row.name for row in sparse],
        "test": [row.name for row in sparse_holdout],
        "laws": {
            law: (
                {
                    "available": True,
                    "heldout_mape": mape(
                        fit_law(sparse, law=law, model_size_field=model_size_field),
                        sparse_holdout,
                        model_size_field,
                    ),
                }
                if len(sparse) > (6 if law == "skaling" else 5)
                else {
                    "available": False,
                    "reason": "insufficient L-shape rows for the fitted parameter count",
                }
            )
            for law in laws
        },
    }
    if include_influence:
        output["influence"] = []
        for heldout in observations:
            train = [row for row in observations if row != heldout]
            row_output: dict[str, object] = {"name": heldout.name, "laws": {}}
            for law in laws:
                fit = fit_law(
                    train,
                    law=law,
                    model_size_field=model_size_field,
                    restarts=16,
                )
                row_output["laws"][law] = {
                    "heldout_mape": mape(fit, [heldout], model_size_field),
                    "alpha": fit.params[2],
                    "beta": fit.params[3],
                    "k": fit.params[4],
                    "E": fit.params[5],
                }
            output["influence"].append(row_output)
    return output


def main() -> None:
    results = [
        evaluate("dense", "total_params"),
        evaluate(
            "dense",
            "total_params",
            minimum_width=32,
            l_shape_width=32,
            current_only=True,
        ),
        evaluate(
            "dense",
            "total_params",
            minimum_width=32,
            minimum_ratio=0.1,
            l_shape_width=32,
            current_only=True,
        ),
        evaluate(
            "dense",
            "total_params",
            minimum_width=64,
            l_shape_width=64,
            include_influence=True,
        ),
        evaluate(
            "dense",
            "total_params",
            minimum_width=64,
            l_shape_width=64,
            current_only=True,
            include_influence=True,
        ),
        evaluate(
            "dense",
            "total_params",
            minimum_width=64,
            minimum_ratio=0.1,
            l_shape_width=64,
            current_only=True,
        ),
        evaluate(
            "moe64a2", "total_params", minimum_width=256, l_shape_ratio=0.01
        ),
        evaluate(
            "moe64a2", "active_params", minimum_width=256, l_shape_ratio=0.01
        ),
        fit_shared_skaling_floor("total_params"),
        fit_shared_skaling_floor("active_params"),
        fit_shared_skaling_floor("total_params", balance_families=True),
        fit_shared_skaling_floor("active_params", balance_families=True),
        fit_moe_with_dense_data_term("total_params"),
        fit_moe_with_dense_data_term("active_params"),
        fit_moe_with_dense_floor("total_params"),
        fit_moe_with_dense_floor("active_params"),
        compare_l_shape_anchors((32, 64), model_size_field="total_params"),
        compare_d32_extrapolation("total_params"),
    ]
    output = HERE / "results.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
