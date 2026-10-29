"""CLI: python -m documentation_agent --program program0"""

from __future__ import annotations

import argparse
import os
import sys

from . import config
from .generate import document_program


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Documentation Agent: write stakeholder markdown for a SAS program.",
    )
    parser.add_argument(
        "--program",
        default="program0",
        help="Program id: program0, program1, or program2 (default: program0)",
    )
    args = parser.parse_args(argv)

    config.get_program(args.program)  # fail fast on a bad id

    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "GOOGLE_API_KEY is not set. Copy .env_template to .env and add a Gemini key.",
            file=sys.stderr,
        )
        return 1

    path = document_program(args.program)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
