# Installation

## Virtual environment

!!! tip
    You will need an environment manager, and I cannot recommend enough [uv](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment with your prefered Python version (3.11 or higher is required to use Holocron):
```bash
$ uv venv --python 3.11
```

=== "Stable"

    ```bash
    $ uv pip install pylocron
    ```

=== "Latest"

    ```bash
    $ uv pip install pylocron @ git+https://github.com/frgfm/holocron.git
    ```


## System installation

You'll need [Python](https://www.python.org/downloads/) 3.11 or higher, and a package installer like [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/).

=== "Stable"

    ```bash
    $ uv pip install --system pylocron
    ```

=== "Latest"

    ```bash
    $ uv pip install --system pylocron @ git+https://github.com/frgfm/holocron.git
    ```

=== "Stable (pip)"

    ```bash
    $ pip install pylocron
    ```

=== "Latest (pip)"

    ```bash
    $ pip install pylocron @ git+https://github.com/frgfm/holocron.git
    ```

!!! info
    Holocron is built on top of [PyTorch](https://github.com/pytorch/pytorch) which is a complex dependency. Proper installation depends on your system and available hardware. You can refer to [installation guide of uv](https://docs.astral.sh/uv/guides/integration/pytorch) which is quite detailed.
