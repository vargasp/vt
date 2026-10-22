# vt

Shared Python library intended for reuse across multiple projects.

## Repository structure

#```text
vt/
├── README.md
├── pyproject.toml
├── environment.yml
├── src/
│   └── vt/
│       ├── __init__.py
│       ├── common_utils.py
│       └── ...
├── tests/
├── data/
└── notebooks/
#```plaintext

## Purpose of folders

- `src/common_lib_a/`: importable library code
- `tests/`: automated tests for the library
- `data/`: small sample or test data that should move with the repo
- `notebooks/`: Jupyter notebooks for exploration and trying new ideas

## Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate vt
```plaintext

## Install the library in editable mode

From the repository root:

```bash
pip install -e .
```plaintext

Editable install means source changes are picked up without reinstalling.

## Verify installation

```bash
python -c "import vt; print(vt.__file__)"
```plaintext

## Run tests

From the repository root:

```bash
pytest
```plaintext

## Spyder setup

This environment includes:

- `spyder-kernels`

If Spyder is installed in a separate `dev-tools` environment, you can still point Spyder to this environment as the interpreter/kernel.

## Jupyter setup

This environment includes:

- `ipykernel`

Register it as a Jupyter kernel:

```bash
python -m ipykernel install --user --name vt --display-name "Python (vt)"
```plaintext

Then select `Python (vt)` inside Jupyter.

## Using this library from a project

Install the library first in the target project environment:

```bash
conda activate project1
pip install -e /path/to/vt
```plaintext

Then project code can import it like:

```python
from vt import common_utils
```plaintext

## Notebook imports

Notebooks should import package code normally, for example:

```python
from vtimport common_utils
```plaintext

Avoid manual path modifications like `sys.path.append(...)`.

## Notes

- Keep reusable logic in `src/vt/`
- Use notebooks for exploration, not as the main home of production code
- Keep only small, appropriate files in `data/`