from pathlib import Path

from chess_engine_4.training.config import load_training_config
from chess_engine_4.training.throughput_sweep import (
    entry_matches,
    load_results,
    make_entry,
    normalize_widths,
    write_results,
)


def test_normalize_widths_sorts_and_deduplicates() -> None:
    assert normalize_widths([128, 32, 128, 64]) == [32, 64, 128]


def test_throughput_result_round_trip(tmp_path: Path) -> None:
    config = load_training_config(Path("configs/dense.py"), d_model=64)
    profile = {
        "warmup_steps": 50,
        "profile_steps": 500,
        "flops_per_sample": 1234,
        "measured_wall_ms_per_step": 10.0,
        "train_gpu": {"mean_ms": 8.0},
        "h2d_copy_gpu": {"mean_ms": 1.0},
        "data_fetch_wall": {"mean_ms": 0.5},
        "gpu_idle_gap_mean_ms": 0.2,
        "train_only_mfu": 0.5,
        "end_to_end_mfu": 0.4,
    }
    entry = make_entry(config, profile, source_commit="abc123")
    path = tmp_path / "throughput.toml"

    write_results(path, {"d64": entry}, config_path=Path("configs/dense.py"))
    loaded = load_results(path)

    assert loaded["d64"] == entry
    assert loaded["d64"]["samples_per_sec"] == 204_800.0
    assert entry_matches(loaded["d64"], config, warmup_steps=50, profile_steps=500)
    assert not entry_matches(loaded["d64"], config, warmup_steps=50, profile_steps=200)
