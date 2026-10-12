"""Defaults retained for existing tournament configurations and result identity."""

from pathlib import Path

# Historical configs omit this path. Keep it stable for resumed tournaments;
# local callers can supply their own book in the tournament configuration.
OPENING_BOOK_PATH = Path("/artifacts/books/UHO_Lichess_4852_v1.pgn")
