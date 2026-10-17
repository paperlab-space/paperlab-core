"""Entry point for the ``paperlab-space`` command."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from paperlab_space._cli import app
    except ImportError as exc:
        if getattr(exc, "name", None) not in ("typer", "rich"):
            raise
        sys.stderr.write(
            "The paperlab-space command needs the CLI extra:\n  pip install 'paperlab-space[cli]'\n"
        )
        raise SystemExit(1) from exc
    app()


if __name__ == "__main__":
    main()
