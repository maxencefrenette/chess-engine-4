from __future__ import annotations

import io
import struct
import tarfile
from pathlib import Path

import numpy as np

from chess_engine_4.data.leela import V6_RECORD_SIZE, iter_records, iter_tar_records


def test_iter_records_parses_v6_record() -> None:
    payload = _v6_record()

    records = list(iter_records(payload))

    assert len(records) == 1
    assert records[0].version == 6
    assert records[0].input_format == 3
    assert records[0].planes.shape == (104, 8, 8)
    assert records[0].policy.shape == (1858,)
    np.testing.assert_allclose(records[0].value, [0.375, 0.5, 0.125])
    assert records[0].visits == 123


def test_iter_tar_records_reads_tar_member(tmp_path: Path) -> None:
    tar_path = tmp_path / "training.tar"
    payload = _v6_record()
    info = tarfile.TarInfo("chunk")
    info.size = len(payload)

    with tarfile.open(tar_path, "w") as tar:
        tar.addfile(info, io.BytesIO(payload))

    records = list(iter_tar_records(tar_path))

    assert len(records) == 1
    assert records[0].version == 6


def _v6_record() -> bytes:
    record = bytearray(V6_RECORD_SIZE)
    struct.pack_into("<II", record, 0, 6, 3)
    policy = np.zeros(1858, dtype="<f4")
    policy[0] = 1.0
    record[8 : 8 + policy.nbytes] = policy.tobytes()
    planes = np.zeros(104, dtype="<u8")
    planes[0] = 1
    record[7440 : 7440 + planes.nbytes] = planes.tobytes()
    struct.pack_into("<fffff", record, 8308, 0.25, 0.5, 0.2, 0.1, 42.0)
    struct.pack_into("<I", record, 8340, 123)
    return bytes(record)
