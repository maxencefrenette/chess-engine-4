from dataclasses import asdict

from chess_engine_4.model import MlpChessNet, MlpChessNetConfig
from chess_engine_4.training.checkpoint2leela import _model_from_checkpoint


def test_loads_dense_checkpoint_with_legacy_policy_config() -> None:
    model = MlpChessNet(MlpChessNetConfig(d_model=32, depth=1))
    config = asdict(model.config)
    config["policy"] = {"kind": "dense"}

    loaded = _model_from_checkpoint(
        {
            "config": {"model": config},
            "model_state_dict": model.state_dict(),
        }
    )

    assert isinstance(loaded, MlpChessNet)
