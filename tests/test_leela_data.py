from __future__ import annotations

import gzip
import io
import math
import struct
import tarfile
from pathlib import Path

import torch

from chess_engine_4.data.leela import (
    COMPACT_POLICY_SIZE,
    HISTORY_PLANE_COUNT,
    POLICY_SIZE,
    V6_RECORD_SIZE,
    LeelaParquetDataset,
    resolve_data_paths,
)
from chess_engine_4.data.native import (
    convert_native_lc0_tar_to_parquet,
    iter_native_packed_batches,
)

POLICY_OFFSET = 8
PLANES_OFFSET = POLICY_OFFSET + POLICY_SIZE * 4
CASTLING_US_OOO_OFFSET = PLANES_OFFSET + HISTORY_PLANE_COUNT * 8
CASTLING_US_OO_OFFSET = CASTLING_US_OOO_OFFSET + 1
CASTLING_THEM_OOO_OFFSET = CASTLING_US_OO_OFFSET + 1
CASTLING_THEM_OO_OFFSET = CASTLING_THEM_OOO_OFFSET + 1
SIDE_TO_MOVE_OFFSET = CASTLING_THEM_OO_OFFSET + 1
RULE50_OFFSET = SIDE_TO_MOVE_OFFSET + 1
ROOT_Q_OFFSET = 8280
BEST_Q_OFFSET = 8284
ROOT_D_OFFSET = 8288
BEST_D_OFFSET = 8292
ROOT_M_OFFSET = 8296
BEST_M_OFFSET = 8300
PLIES_LEFT_OFFSET = 8304
RESULT_Q_OFFSET = 8308
RESULT_D_OFFSET = 8312
PLAYED_Q_OFFSET = 8316
PLAYED_D_OFFSET = 8320
PLAYED_M_OFFSET = 8324
ORIG_Q_OFFSET = 8328
ORIG_D_OFFSET = 8332
ORIG_M_OFFSET = 8336


def test_native_tar_decoder_yields_tensor_batches_from_gzip_members(
    tmp_path: Path,
) -> None:
    tar_path = tmp_path / "training.tar"
    _write_tar(tar_path, gzip.compress(_records(4)))

    batches = list(_tar_batches(tar_path, batch_size=2))

    assert len(batches) == 2
    packed_planes, plane_scalars, policy_indices, policy_probs, value = batches[0]
    assert tuple(packed_planes.shape) == (2, 104, 8)
    assert tuple(plane_scalars.shape) == (2, 8)
    assert tuple(policy_indices.shape) == (2, COMPACT_POLICY_SIZE)
    assert tuple(policy_probs.shape) == (2, COMPACT_POLICY_SIZE)
    assert tuple(value.shape) == (2, 6, 3)
    assert packed_planes.dtype == torch.uint8
    assert plane_scalars.dtype == torch.float32
    assert policy_indices.dtype == torch.int16
    assert policy_probs.dtype == torch.float16
    assert value.dtype == torch.float32

    assert packed_planes[0, 0, 0].item() == 0x80
    torch.testing.assert_close(
        plane_scalars[0],
        torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 50.0, 0.0, 1.0]),
    )
    assert policy_indices[0, :3].tolist() == [0, 2, -1]
    torch.testing.assert_close(
        policy_probs[0, :3],
        torch.tensor([0.75, 0.25, 0.0], dtype=torch.float16),
    )
    torch.testing.assert_close(value[0, 0], torch.tensor([1.0, 0.0, 42.0]))
    torch.testing.assert_close(value[0, 1], torch.tensor([0.5, 0.25, 12.0]))
    torch.testing.assert_close(value[0, 4], torch.tensor([0.75, 0.125, 20.0]))
    torch.testing.assert_close(value[0, 5, :2], torch.tensor([0.0, 0.0]))
    assert torch.isnan(value[0, 5, 2])
    assert tuple(batches[1][0].shape) == (2, 104, 8)


def test_native_tar_decoder_drops_incomplete_final_batch(tmp_path: Path) -> None:
    tar_path = tmp_path / "training.tar"
    _write_tar(tar_path, gzip.compress(_records(3)))

    batches = list(_tar_batches(tar_path, batch_size=2))

    assert len(batches) == 1
    assert tuple(batches[0][0].shape) == (2, 104, 8)


def test_parquet_conversion_preserves_training_inputs_and_root_targets(tmp_path: Path) -> None:
    tar_path = tmp_path / "training.tar"
    parquet_path = tmp_path / "training.parquet"
    _write_tar(tar_path, gzip.compress(_records(4)))

    records, input_bytes, output_bytes = convert_native_lc0_tar_to_parquet(tar_path, parquet_path)
    tar_batch = next(iter(_tar_batches(tar_path, batch_size=4)))
    parquet_batch = next(iter(LeelaParquetDataset(parquet_path, batch_size=4, threads=1)))

    assert records == 4
    assert input_bytes == tar_path.stat().st_size
    assert output_bytes == parquet_path.stat().st_size
    for tar_tensor, parquet_tensor in zip(tar_batch[:4], parquet_batch[:4], strict=True):
        torch.testing.assert_close(tar_tensor, parquet_tensor, rtol=0, atol=0)
    torch.testing.assert_close(tar_batch[4][:, 4], parquet_batch[4][:, 4], rtol=0, atol=0)


def test_leela_parquet_dataset_drops_incomplete_final_batch(tmp_path: Path) -> None:
    tar_path = tmp_path / "training.tar"
    parquet_path = tmp_path / "training.parquet"
    _write_tar(tar_path, gzip.compress(_records(3)))
    convert_native_lc0_tar_to_parquet(tar_path, parquet_path)

    batches = list(LeelaParquetDataset(parquet_path, batch_size=2, threads=1))

    assert len(batches) == 1
    assert tuple(batches[0][0].shape) == (2, 104, 8)


def test_resolve_data_paths_loads_dotenv(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parquet_path = data_dir / "training.parquet"
    parquet_path.touch()
    env_path = tmp_path / ".env"
    env_path.write_text(f"CHESS_ENGINE_4_DATA_PATH={data_dir}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHESS_ENGINE_4_DATA_PATH", raising=False)

    assert resolve_data_paths(None) == [parquet_path]


def _tar_batches(path: Path, *, batch_size: int):
    return iter_native_packed_batches(
        [path],
        batch_size=batch_size,
        prefetch_per_thread=2,
        threads=1,
    )


def _write_tar(path: Path, payload: bytes) -> None:
    info = tarfile.TarInfo("training.1.gz")
    info.size = len(payload)
    with tarfile.open(path, "w") as tar:
        tar.addfile(info, io.BytesIO(payload))


def _records(count: int) -> bytes:
    records = bytearray(count * V6_RECORD_SIZE)
    for index in range(count):
        start = index * V6_RECORD_SIZE
        record = memoryview(records)[start : start + V6_RECORD_SIZE]
        struct.pack_into("<I", record, 0, 6)
        struct.pack_into("<I", record, 4, 1)
        for policy_index in range(POLICY_SIZE):
            struct.pack_into("<f", record, POLICY_OFFSET + policy_index * 4, -1.0)
        struct.pack_into("<f", record, POLICY_OFFSET, 0.75)
        struct.pack_into("<f", record, POLICY_OFFSET + 2 * 4, 0.25)
        struct.pack_into("<Q", record, PLANES_OFFSET, 0x80)
        record[CASTLING_US_OOO_OFFSET] = 1
        record[CASTLING_US_OO_OFFSET] = 0
        record[CASTLING_THEM_OOO_OFFSET] = 1
        record[CASTLING_THEM_OO_OFFSET] = 0
        record[SIDE_TO_MOVE_OFFSET] = 1
        record[RULE50_OFFSET] = 50
        _pack_f32(record, ROOT_Q_OFFSET, 0.75)
        _pack_f32(record, BEST_Q_OFFSET, 0.5)
        _pack_f32(record, ROOT_D_OFFSET, 0.125)
        _pack_f32(record, BEST_D_OFFSET, 0.25)
        _pack_f32(record, ROOT_M_OFFSET, 20.0)
        _pack_f32(record, BEST_M_OFFSET, 12.0)
        _pack_f32(record, PLIES_LEFT_OFFSET, 42.0)
        _pack_f32(record, RESULT_Q_OFFSET, 1.0)
        _pack_f32(record, RESULT_D_OFFSET, 0.0)
        _pack_f32(record, PLAYED_Q_OFFSET, 0.25)
        _pack_f32(record, PLAYED_D_OFFSET, 0.5)
        _pack_f32(record, PLAYED_M_OFFSET, 8.0)
        _pack_f32(record, ORIG_Q_OFFSET, math.nan)
        _pack_f32(record, ORIG_D_OFFSET, math.nan)
        _pack_f32(record, ORIG_M_OFFSET, math.nan)
    return bytes(records)


def _pack_f32(record: memoryview, offset: int, value: float) -> None:
    struct.pack_into("<f", record, offset, value)
