# Installation

!!! info "Choose the matching version"
    PyPI `0.3.0` includes the APIs documented here. Development installations
    follow the `main` branch and include subsequent changes.

## Virtual environment

!!! tip
    You will need an environment manager, and I cannot recommend enough [uv](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment with your preferred Python version (3.11 or higher is required to use Holocron):
```bash
$ uv venv --python 3.11
```

=== "Development (main branch)"

    ```bash
    $ uv pip install "pylocron @ git+https://github.com/frgfm/holocron.git"
    ```

=== "Stable v0.3.0"

    ```bash
    $ uv pip install pylocron==0.3.0
    ```

## System installation

You'll need [Python](https://www.python.org/downloads/) 3.11 or higher, and a package installer like [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/).

=== "Development (uv)"

    ```bash
    $ uv pip install --system "pylocron @ git+https://github.com/frgfm/holocron.git"
    ```

=== "Stable v0.3.0 (uv)"

    ```bash
    $ uv pip install --system pylocron==0.3.0
    ```

=== "Development (pip)"

    ```bash
    $ pip install "pylocron @ git+https://github.com/frgfm/holocron.git"
    ```

=== "Stable v0.3.0 (pip)"

    ```bash
    $ pip install pylocron==0.3.0
    ```

!!! info
    Holocron is built on top of [PyTorch](https://github.com/pytorch/pytorch) which is a complex dependency. Proper installation depends on your system and available hardware. You can refer to [installation guide of uv](https://docs.astral.sh/uv/guides/integration/pytorch) which is quite detailed.
