from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import torch

from chess_engine_4.data.leela import (
    LEELA_V6_DTYPE,
    V6_RECORD_SIZE,
    LeelaTarDataset,
    parse_frame_chunk,
    planes_from_frames,
    resolve_data_paths,
    tensors_from_frames,
)


def test_parse_frame_chunk_reads_v6_records() -> None:
    payload = _records(2)

    frames = parse_frame_chunk(payload)

    assert len(frames) == 2
    assert frames.dtype == LEELA_V6_DTYPE
    assert frames["version"].tolist() == [6, 6]
    assert frames["input_format"].tolist() == [1, 1]


def test_tensors_from_frames_matches_lc0_shapes_and_fields() -> None:
    frames = parse_frame_chunk(_records(2))

    batch = tensors_from_frames(frames)

    assert tuple(batch.planes.shape) == (2, 112, 8, 8)
    assert tuple(batch.policy.shape) == (2, 1858)
    assert tuple(batch.value.shape) == (2, 6, 3)
    assert batch.planes.dtype == torch.float32
    assert batch.policy.dtype == torch.float32
    assert batch.value.dtype == torch.float32

    assert batch.planes[0, 0, 0, 0].item() == 1.0
    assert batch.planes[0, 0, 0, 1].item() == 0.0
    assert torch.all(batch.planes[:, 111] == 1.0)
    assert torch.all(batch.planes[:, 110] == 0.0)

    assert batch.policy[0, 0].item() == 1.0
    assert batch.policy[0, 1].item() == -1.0
    np.testing.assert_allclose(batch.value[0, 0].numpy(), [1.0, 0.0, 42.0])
    np.testing.assert_allclose(batch.value[0, 1].numpy(), [0.5, 0.25, 12.0])
    np.testing.assert_allclose(batch.value[0, 5, :2].numpy(), [0.0, 0.0])
    assert torch.isnan(batch.value[0, 5, 2])


def test_tensors_from_frames_supports_compact_planes() -> None:
    frames = parse_frame_chunk(_records(2))

    compact = tensors_from_frames(frames, compact_planes=True)
    dense_planes = planes_from_frames(frames)

    assert tuple(compact.binary_planes.shape) == (2, 104, 8, 8)
    assert tuple(compact.plane_scalars.shape) == (2, 8)
    assert compact.binary_planes.dtype == torch.uint8
    assert compact.plane_scalars.dtype == torch.float32
    np.testing.assert_array_equal(compact.binary_planes.numpy(), dense_planes[:, :104])
    scalar_planes = np.broadcast_to(
        compact.plane_scalars.numpy()[:, :, None, None],
        dense_planes[:, 104:].shape,
    )
    np.testing.assert_allclose(scalar_planes, dense_planes[:, 104:])


def test_leela_tar_dataset_yields_tensor_batches_from_gzip_members(tmp_path: Path) -> None:
    tar_path = tmp_path / "training.tar"
    payload = gzip.compress(_records(3))
    info = tarfile.TarInfo("training.1.gz")
    info.size = len(payload)

    with tarfile.open(tar_path, "w") as tar:
        tar.addfile(info, io.BytesIO(payload))

    batches = list(LeelaTarDataset(tar_path, batch_size=2))

    assert len(batches) == 2
    assert tuple(batches[0][0].shape) == (2, 112, 8, 8)
    assert tuple(batches[1][0].shape) == (1, 112, 8, 8)


def test_leela_tar_dataset_can_yield_compact_plane_batches(tmp_path: Path) -> None:
    tar_path = tmp_path / "training.tar"
    payload = gzip.compress(_records(3))
    info = tarfile.TarInfo("training.1.gz")
    info.size = len(payload)

    with tarfile.open(tar_path, "w") as tar:
        tar.addfile(info, io.BytesIO(payload))

    batches = list(LeelaTarDataset(tar_path, batch_size=2, compact_planes=True))

    assert len(batches) == 2
    assert tuple(batches[0][0].shape) == (2, 104, 8, 8)
    assert tuple(batches[0][1].shape) == (2, 8)
    assert tuple(batches[0][2].shape) == (2, 1858)


def test_resolve_data_paths_loads_dotenv(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path = data_dir / "training.tar"
    tar_path.touch()
    env_path = tmp_path / ".env"
    env_path.write_text(f"CHESS_ENGINE_4_DATA_PATH={data_dir}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHESS_ENGINE_4_DATA_PATH", raising=False)

    assert resolve_data_paths(None) == [tar_path]


def _records(count: int) -> bytes:
    records = np.zeros(count, dtype=LEELA_V6_DTYPE)
    records["version"] = 6
    records["input_format"] = 1
    records["probabilities"].fill(-1.0)
    records["probabilities"][:, 0] = 1.0
    records["planes"][:, 0] = 0x80
    records["castling_us_ooo"] = 1
    records["castling_us_oo"] = 0
    records["castling_them_ooo"] = 1
    records["castling_them_oo"] = 0
    records["side_to_move_or_enpassant"] = 1
    records["rule50_count"] = 50
    records["root_q"] = 0.75
    records["best_q"] = 0.5
    records["root_d"] = 0.125
    records["best_d"] = 0.25
    records["root_m"] = 20.0
    records["best_m"] = 12.0
    records["plies_left"] = 42.0
    records["result_q"] = 1.0
    records["result_d"] = 0.0
    records["played_q"] = 0.25
    records["played_d"] = 0.5
    records["played_m"] = 8.0
    records["orig_q"] = np.nan
    records["orig_d"] = np.nan
    records["orig_m"] = np.nan
    records["visits"] = 123
    records["played_idx"] = 0
    records["best_idx"] = 0
    records["policy_kld"] = 0.01
    payload = records.tobytes()
    assert len(payload) == count * V6_RECORD_SIZE
    return payload
