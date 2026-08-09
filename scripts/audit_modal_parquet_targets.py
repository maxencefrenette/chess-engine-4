"""Sample root target distributions from every canonical Modal Parquet shard."""

from __future__ import annotations

import math
from pathlib import Path

from chess_engine_4.modal_train import (
    REMOTE_DATA_PATH,
    REMOTE_PARQUET_DATA_PATH,
    app,
    data_volume,
    image,
)

ROWS_PER_SHARD = 256
SHARDS_PER_TASK = 10


@app.function(
    image=image,
    cpu=4,
    max_containers=8,
    retries=2,
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=30 * 60,
)
def audit_chunk(names: list[str]) -> dict[str, float | int]:
    import numpy as np

    from chess_engine_4.data.native import iter_native_parquet_batches

    result: dict[str, float | int] = {
        "rows": 0,
        "q_sum": 0.0,
        "q_sq": 0.0,
        "q_min": math.inf,
        "q_max": -math.inf,
        "d_sum": 0.0,
        "d_sq": 0.0,
        "d_min": math.inf,
        "d_max": -math.inf,
        "m_sum": 0.0,
        "m_sq": 0.0,
        "m_min": math.inf,
        "m_max": -math.inf,
        "nonfinite": 0,
        "invalid_q": 0,
        "invalid_d": 0,
        "invalid_wdl": 0,
        "negative_m": 0,
    }
    for name in names:
        batch = next(
            iter_native_parquet_batches(
                [Path(REMOTE_PARQUET_DATA_PATH) / name],
                batch_size=ROWS_PER_SHARD,
                prefetch_per_thread=1,
                threads=1,
            )
        )
        root_targets = batch[4][:, 4].numpy()
        q = root_targets[:, 0]
        d = root_targets[:, 1]
        m = root_targets[:, 2]
        finite = np.isfinite(q) & np.isfinite(d) & np.isfinite(m)
        result["rows"] += len(q)
        result["nonfinite"] += int((~finite).sum())
        result["invalid_q"] += int(((q < -1.00001) | (q > 1.00001)).sum())
        result["invalid_d"] += int(((d < -0.00001) | (d > 1.00001)).sum())
        result["invalid_wdl"] += int((np.abs(q) > 1.0 - d + 0.00002).sum())
        result["negative_m"] += int((m < 0).sum())
        for key, values in (("q", q), ("d", d), ("m", m)):
            values_f64 = values.astype(np.float64)
            result[f"{key}_sum"] += float(values_f64.sum())
            result[f"{key}_sq"] += float(np.square(values_f64).sum())
            result[f"{key}_min"] = min(
                float(result[f"{key}_min"]), float(values.min())
            )
            result[f"{key}_max"] = max(
                float(result[f"{key}_max"]), float(values.max())
            )
    return result


def main() -> None:
    names = sorted(
        Path(entry.path).name
        for entry in data_volume.listdir("/parquet")
        if entry.type == 1 and entry.path.endswith(".parquet")
    )
    chunks = [
        names[offset : offset + SHARDS_PER_TASK]
        for offset in range(0, len(names), SHARDS_PER_TASK)
    ]
    results: list[dict[str, float | int]] = []
    with app.run():
        results.extend(audit_chunk.map(chunks, order_outputs=False))

    summed_keys = (
        "rows",
        "q_sum",
        "q_sq",
        "d_sum",
        "d_sq",
        "m_sum",
        "m_sq",
        "nonfinite",
        "invalid_q",
        "invalid_d",
        "invalid_wdl",
        "negative_m",
    )
    totals = {
        key: sum(float(result[key]) for result in results) for key in summed_keys
    }
    for key in ("q_min", "d_min", "m_min"):
        totals[key] = min(float(result[key]) for result in results)
    for key in ("q_max", "d_max", "m_max"):
        totals[key] = max(float(result[key]) for result in results)

    sampled_rows = totals["rows"]
    print(
        f"target_audit shards={len(names)} sampled_rows={int(sampled_rows)} "
        f"nonfinite={int(totals['nonfinite'])} "
        f"invalid_q={int(totals['invalid_q'])} "
        f"invalid_d={int(totals['invalid_d'])} "
        f"invalid_wdl={int(totals['invalid_wdl'])} "
        f"negative_m={int(totals['negative_m'])}"
    )
    for key in ("q", "d", "m"):
        mean = totals[f"{key}_sum"] / sampled_rows
        variance = totals[f"{key}_sq"] / sampled_rows - mean * mean
        print(
            f"root_{key} mean={mean:.9f} std={math.sqrt(max(0.0, variance)):.9f} "
            f"min={totals[f'{key}_min']:.9f} max={totals[f'{key}_max']:.9f}"
        )


if __name__ == "__main__":
    main()
