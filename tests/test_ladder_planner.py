from pathlib import Path

from chess_engine_4.training.ladder_planner import read_ladder


def test_dense_ladder_is_the_declared_l_shape() -> None:
    ladder = read_ladder(Path("experiments/scaling-ladders.toml"), "dense")

    assert ladder.anchor_width == 64
    assert ladder.width_ratio == 0.055
    assert len(ladder.scaffold) == len(ladder.widths) + len(ladder.data_ratios) - 1
    assert len(ladder.grid) == len(ladder.widths) * len(ladder.data_ratios)
