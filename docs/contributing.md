# Contributing

How to set up the project, run the checks, and change the code or the docs.

Bug reports, fixes, new metrics, and documentation improvements are all welcome. For anything larger than a small fix, open an issue first so we can agree on the approach.

## Set up

The project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/paperlab-space/paperlab-core
cd paperlab-core
uv sync
```

## Run the checks

CI runs all of these on Python 3.10 to 3.13:

```bash
uv run pytest                   # unit tests, no network
uv run ruff check .
uv run ruff format --check .    # also formats Python blocks in docs/*.md
uv run mypy src tests
uv run python scripts/build_docs.py --check
```

Unit tests never call a real provider. To run the smoke tests against real APIs, set the keys and run:

```bash
uv run pytest -m live
```

## Code guidelines

- Every public function and class is typed (`mypy --strict`) and has a docstring. Public fields have attribute docstrings: a string on the line after the field.
- The core never imports web code: no database, auth, or payments. It is input, evaluation, output.
- Error messages say what is wrong and how to fix it, in plain words. Raise `DataError`, `ConfigError`, or `TemplateError`, not bare exceptions.
- Anything that changes outputs for the same config (a metric's behaviour, a default) is a breaking change and needs a note in the [changelog](changelog.md).

## Change the docs

The docs are the Markdown files in `docs/`, published at [paperlab.space/docs](https://paperlab.space/docs).

- **Guides and other pages:** edit the `.md` file. Link to other pages with relative paths (`guides/data.md`), so links work on GitHub and on the site.
- **New pages:** add the file to `docs/nav.json`, which sets the sidebar. A page's first `#` heading is its title, and the paragraph after it is its description.
- **Reference pages** (`docs/reference/config.md`, `metrics.md`, and `python.md`) are generated from the source code. Edit the docstrings, then run `uv run python scripts/build_docs.py`. A new public export must be added to the `API` list in that script, or it fails.

The site is rebuilt from `main` when the platform deploys.

## Release

1. Update `__version__` in `src/paperlab_space/__init__.py` and add a section for the new version to the [changelog](changelog.md).
2. Merge to `main`.
3. Publish a GitHub release tagged `vX.Y.Z`. The publish workflow checks the tag against the version, runs the tests, and uploads to PyPI. Zenodo archives the release and mints its DOI.
