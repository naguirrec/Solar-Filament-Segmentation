"""Export a training checkpoint to an fp16 file under weights/ and update the manifest.

Usage (needs torch)::

    uv run --extra gpu python scripts/export_weights.py \\
        path/to/unet_r34_crops.pt --name unet_r34_crops_fp16.pt

It writes ``weights/<name>`` and ``weights/MANIFEST.json`` (file SHA-256, tensor
SHA-256 of the fp16 state and of the original, size, encoder, epochs). It fails,
and writes nothing, when the file is over 100 MB. It does not commit anything:
the weights are not committed without the owner's OK (``AGENTS.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.weights import (
    R34_FP16_FILE,
    WEIGHTS_DIR_NAME,
    WeightsError,
    export_weights,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="a training checkpoint (.pt)")
    parser.add_argument(
        "--name",
        default=None,
        help=f"output file name (default <checkpoint stem>_fp16.pt, e.g. {R34_FP16_FILE})",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=ROOT / WEIGHTS_DIR_NAME,
        help="output directory (default: weights/ of the repo)",
    )
    args = parser.parse_args(argv)
    try:
        entry = export_weights(args.checkpoint, args.weights_dir, file_name=args.name)
    except WeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=2, sort_keys=True))
    print(f"wrote {args.weights_dir} (and MANIFEST.json). Nothing was committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
