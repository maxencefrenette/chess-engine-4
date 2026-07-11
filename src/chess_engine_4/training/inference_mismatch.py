"""Compare native training inference with lc0 ONNX inference."""

from __future__ import annotations

import gzip
import random
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from chess_engine_4.data.leela import (
    BOARD_SIZE,
    HISTORY_PLANE_COUNT,
    V6_RECORD_SIZE,
)
from chess_engine_4.model import build_model, model_config_from_dict
from chess_engine_4.model.transformer_engine import autocast_context
from chess_engine_4.training.inference_comparison import NetworkOutputs
from chess_engine_4.training.packed_input import PlaneInputExpander

_PLANES_OFFSET = 8 + 1858 * 4
_PACKED_PLANES_SIZE = HISTORY_PLANE_COUNT * BOARD_SIZE
_CASTLING_OFFSET = _PLANES_OFFSET + _PACKED_PLANES_SIZE
_SIDE_TO_MOVE_OFFSET = _CASTLING_OFFSET + 4
_RULE50_OFFSET = _SIDE_TO_MOVE_OFFSET + 1
_HISTORY_LENGTH = 8


@dataclass(slots=True)
class SampledPosition:
    initial_fen: str
    moves: tuple[str, ...]
    packed_planes: np.ndarray
    plane_scalars: np.ndarray


def sample_training_positions(
    paths: list[Path],
    *,
    count: int,
    seed: int,
) -> list[SampledPosition]:
    """Sample positions while retaining enough game history for lc0's input encoder."""

    rng = random.Random(seed)
    shuffled_paths = list(paths)
    rng.shuffle(shuffled_paths)
    samples: list[SampledPosition] = []
    for path in shuffled_paths:
        with tarfile.open(path, mode="r:*") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            rng.shuffle(members)
            for member in members:
                if Path(member.name).name == "LICENSE":
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                if payload.startswith(b"\x1f\x8b"):
                    payload = gzip.decompress(payload)
                if len(payload) % V6_RECORD_SIZE:
                    raise ValueError(
                        f"{path}:{member.name} has {len(payload)} bytes, "
                        f"not a multiple of {V6_RECORD_SIZE}"
                    )
                records = [
                    payload[offset : offset + V6_RECORD_SIZE]
                    for offset in range(0, len(payload), V6_RECORD_SIZE)
                ]
                samples.extend(_positions_from_game(records, count - len(samples)))
                if len(samples) >= count:
                    return samples
    raise ValueError(f"Only found {len(samples)} usable positions; requested {count}.")


def evaluate_native_checkpoint(
    checkpoint_path: Path,
    positions: list[SampledPosition],
    *,
    batch_size: int = 128,
) -> NetworkOutputs:
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint does not contain config.model.")
    precision = config.get("precision", {}).get("recipe", "mxfp8")
    model = build_model(model_config_from_dict(config["model"])).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    expander = PlaneInputExpander().cuda().eval()

    policies: list[dict[int, float]] = []
    q_parts: list[torch.Tensor] = []
    d_parts: list[torch.Tensor] = []
    with torch.no_grad(), autocast_context(precision):
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            actual_batch_size = len(batch)
            padded_batch = batch + [batch[-1]] * (-actual_batch_size % 32)
            packed = torch.from_numpy(
                np.stack([item.packed_planes for item in padded_batch])
            ).cuda()
            scalars = torch.from_numpy(
                np.stack([item.plane_scalars for item in padded_batch])
            ).to(
                device="cuda", dtype=torch.bfloat16
            )
            output = model(expander(packed, scalars))
            wdl = torch.softmax(output.wdl_logits[:actual_batch_size].float(), dim=-1)
            q_parts.append((wdl[:, 0] - wdl[:, 2]).cpu())
            d_parts.append(wdl[:, 1].cpu())
            logits = output.policy_logits[:actual_batch_size].float().cpu()
            # Native policies are restricted to the same legal indices lc0 reports below.
            policies.extend(logits.numpy())

    return NetworkOutputs(
        policies=policies,
        q=torch.cat(q_parts).numpy(),
        d=torch.cat(d_parts).numpy(),
    )


def _positions_from_game(records: list[bytes], limit: int) -> list[SampledPosition]:
    if limit <= 0 or len(records) <= _HISTORY_LENGTH:
        return []
    boards = [_board_from_record(record) for record in records]
    initial_fen = boards[0].fen(en_passant="fen")
    board = boards[0]
    moves: list[str] = []
    positions: list[SampledPosition] = []
    for index, (record, target) in enumerate(zip(records[1:], boards[1:], strict=True), start=1):
        matching_moves = []
        for move in board.legal_moves:
            candidate = board.copy(stack=False)
            candidate.push(move)
            if _same_position(candidate, target):
                matching_moves.append(move)
        if len(matching_moves) != 1:
            return []
        move = matching_moves[0]
        board.push(move)
        moves.append(move.uci())
        if index >= _HISTORY_LENGTH - 1:
            positions.append(
                SampledPosition(
                    initial_fen=initial_fen,
                    moves=tuple(moves),
                    packed_planes=np.frombuffer(
                        record,
                        dtype=np.uint8,
                        count=_PACKED_PLANES_SIZE,
                        offset=_PLANES_OFFSET,
                    )
                    .reshape(HISTORY_PLANE_COUNT, BOARD_SIZE)
                    .copy(),
                    plane_scalars=_plane_scalars(record),
                )
            )
            if len(positions) >= limit:
                break
    return positions


def _board_from_record(record: bytes) -> chess.Board:
    version, input_format = struct.unpack_from("<II", record)
    if version != 6 or input_format != 1:
        raise ValueError(
            f"Expected v6 classical input, got version={version}, format={input_format}."
        )
    packed = np.frombuffer(
        record,
        dtype=np.uint8,
        count=_PACKED_PLANES_SIZE,
        offset=_PLANES_OFFSET,
    ).reshape(HISTORY_PLANE_COUNT, BOARD_SIZE)
    planes = np.unpackbits(packed, axis=1, bitorder="big").reshape(HISTORY_PLANE_COUNT, 64)
    black_to_move = record[_SIDE_TO_MOVE_OFFSET] != 0
    board = chess.Board.empty()
    board.turn = not black_to_move
    piece_types = (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    )
    for relative_color, base in ((True, 0), (False, 6)):
        absolute_color = relative_color if not black_to_move else not relative_color
        for piece_offset, piece_type in enumerate(piece_types):
            for square in np.flatnonzero(planes[base + piece_offset]):
                absolute_square = int(square) if not black_to_move else int(square) ^ 56
                board.set_piece_at(absolute_square, chess.Piece(piece_type, absolute_color))

    us_ooo, us_oo, them_ooo, them_oo = record[_CASTLING_OFFSET : _CASTLING_OFFSET + 4]
    if black_to_move:
        white_ooo, white_oo, black_ooo, black_oo = them_ooo, them_oo, us_ooo, us_oo
    else:
        white_ooo, white_oo, black_ooo, black_oo = us_ooo, us_oo, them_ooo, them_oo
    board.castling_rights = 0
    for enabled, square in (
        (white_ooo, chess.A1),
        (white_oo, chess.H1),
        (black_ooo, chess.A8),
        (black_oo, chess.H8),
    ):
        if enabled:
            board.castling_rights |= chess.BB_SQUARES[square]

    board.ep_square = _en_passant_square(planes, black_to_move)
    board.halfmove_clock = record[_RULE50_OFFSET]
    board.fullmove_number = 1
    return board


def _en_passant_square(planes: np.ndarray, black_to_move: bool) -> int | None:
    current = planes[6]
    previous = planes[13 + 6]
    difference = np.flatnonzero(current != previous)
    if len(difference) != 2 or not previous.any():
        return None
    old_square = int(np.flatnonzero(previous & (current != previous))[0])
    new_square = int(np.flatnonzero(current & (current != previous))[0])
    if old_square % 8 != new_square % 8 or abs(old_square // 8 - new_square // 8) != 2:
        return None
    rank = 2 if black_to_move else 5
    return chess.square(new_square % 8, rank)


def _plane_scalars(record: bytes) -> np.ndarray:
    return np.asarray(
        [
            *record[_CASTLING_OFFSET : _CASTLING_OFFSET + 4],
            record[_SIDE_TO_MOVE_OFFSET],
            record[_RULE50_OFFSET] / 99.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )


def _same_position(left: chess.Board, right: chess.Board) -> bool:
    return (
        left.board_fen() == right.board_fen()
        and left.turn == right.turn
        and left.castling_rights == right.castling_rights
        and left.ep_square == right.ep_square
        and left.halfmove_clock == right.halfmove_clock
    )

