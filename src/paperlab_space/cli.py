"""Entry point for the ``paperlab-space`` command."""

from __future__ import annotations


def main() -> None:
    # Imported here so `import paperlab_space` does not load the CLI libraries.
    from paperlab_space._cli import app

    app()


if __name__ == "__main__":
    main()
