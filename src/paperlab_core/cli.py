"""Entry point for the ``paperlab`` command."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from paperlab_core._cli import app
    except ImportError as exc:
        if getattr(exc, "name", None) not in ("typer", "rich"):
            raise
        sys.stderr.write(
            "The paperlab command needs the CLI extra:\n  pip install 'paperlab-core[cli]'\n"
        )
        raise SystemExit(1) from exc
    app()


if __name__ == "__main__":
    main()
