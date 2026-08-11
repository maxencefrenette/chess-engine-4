from pathlib import Path

from chess_engine_4.training.ladder_planner import LadderSpec, missing_scaffold, read_ladder


def test_dense_ladder_is_the_declared_l_shape() -> None:
    ladder = read_ladder(Path("experiments/scaling-ladders.toml"), "dense")

    assert ladder.anchor_width == 64
    assert ladder.width_ratio == 0.055
    assert len(ladder.scaffold) == len(ladder.widths) + len(ladder.data_ratios) - 1
    assert len(ladder.grid) == len(ladder.widths) * len(ladder.data_ratios)


def test_spiked_observation_still_covers_scaffold_coordinate() -> None:
    ladder = LadderSpec(
        family="dense",
        anchor_width=64,
        width_ratio=0.055,
        widths=(64, 128),
        data_ratios=(0.055, 0.1),
    )
    observed_including_spike = frozenset({(64, 0.055), (64, 0.1), (128, 0.055)})

    assert missing_scaffold(ladder, observed_including_spike) == []
