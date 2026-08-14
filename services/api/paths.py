from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PAIRS_DIR = DATA_DIR / "pairs"
STYLES_DIR = DATA_DIR / "styles"
OUTPUTS_DIR = DATA_DIR / "outputs"
REFERENCE_PDFS = REPO_ROOT / "reference-pdfs"
STATE_FILE = DATA_DIR / "state.json"

INK_BLUE = (0x25, 0x63, 0xEB)  # #2563EB

# Printable ASCII allowed for Emuru / ByT5 generation (after unicode normalize).
EMURU_ALLOWED = frozenset(chr(c) for c in range(32, 127))

# Default transcription for seeded representative_text.png (densest-line crop).
DEFAULT_STYLE_TEXT = "An erratic some-CAPS"

# Legacy aliases (charset no longer used for hard filtering beyond printable ASCII)
ONEDM_LETTERS = "".join(sorted(EMURU_ALLOWED))
DIFFBRUSH_LETTERS = ONEDM_LETTERS
