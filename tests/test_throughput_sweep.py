from pathlib import Path

from chess_engine_4.training.config import load_training_config
from chess_engine_4.training.throughput_sweep import (
    _benchmark_config,
    _results_gpu,
    entry_matches,
    load_results,
    make_entry,
    normalize_widths,
    write_results,
)


def test_normalize_widths_sorts_and_deduplicates() -> None:
    assert normalize_widths([128, 64, 128, 256]) == [64, 128, 256]


def test_half_batch_benchmark_preserves_recipe_samples() -> None:
    original = load_training_config(Path("configs/dense.py"), d_model=512, training_ratio=1.0)

    half_batch = _benchmark_config(
        original,
        gpu=None,
        batch_size=None,
        batch_divisor=2,
        quantization_recipe=None,
        kernel_backend=None,
    )

    assert half_batch.run.batch_size == original.run.batch_size // 2
    assert half_batch.run.steps == original.run.steps * 2
    assert half_batch.run.batch_size * half_batch.run.steps == (
        original.run.batch_size * original.run.steps
    )


def test_cached_sweep_gpu_metadata_uses_all_rows() -> None:
    assert _results_gpu({"d64": {"gpu": "RTX-PRO-6000"}, "d512": {"gpu": "B200"}}) == (
        "mixed"
    )


def test_throughput_result_round_trip(tmp_path: Path) -> None:
    config = load_training_config(Path("configs/dense.py"), d_model=64)
    profile = {
        "warmup_steps": 50,
        "profile_steps": 500,
        "flops_per_sample": 1234,
        "measured_wall_ms_per_step": 10.0,
        "train_gpu": {"mean_ms": 8.0},
        "pin_memory_wall": {"mean_ms": 0.3},
        "h2d_enqueue_wall": {"mean_ms": 0.1},
        "h2d_copy_gpu": {"mean_ms": 1.0},
        "data_fetch_wall": {"mean_ms": 0.5},
        "gpu_idle_gap_mean_ms": 0.2,
        "peak_memory_allocated_bytes": 1_000_000,
        "peak_memory_reserved_bytes": 2_000_000,
        "train_only_mfu": 0.5,
        "end_to_end_mfu": 0.4,
    }
    entry = make_entry(config, profile, source_commit="abc123")
    path = tmp_path / "throughput.toml"

    write_results(
        path,
        {"d64": entry},
        config_path=Path("configs/dense.py"),
        model_family="dense",
        gpu="B200",
    )
    loaded = load_results(path)

    assert loaded["d64"] == entry
    assert loaded["d64"]["samples_per_sec"] == 204_800.0
    assert entry_matches(loaded["d64"], config, warmup_steps=50, profile_steps=500)
    assert not entry_matches(loaded["d64"], config, warmup_steps=50, profile_steps=200)
