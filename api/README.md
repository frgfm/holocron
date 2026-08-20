# Template for your Vision API using Holocron

## Installation

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [uv](https://docs.astral.sh/uv/), and optionally [Docker](https://docs.docker.com/get-docker/). The locked environment installs the remaining dependencies.

## Usage

### Starting your web server

You will need to clone the repository first:
```shell
git clone https://github.com/frgfm/Holocron.git
cd Holocron
```
Then start the development server from the repository root:

```shell
make lock-backend
make uvicorn-backend
```
Once completed, your [FastAPI](https://fastapi.tiangolo.com/) server should be running on port 8080.

To build and run the container instead, use `make start-backend`. Run the API tests with `make test-backend`.

### Documentation and swagger

FastAPI comes with many advantages including speed and OpenAPI features. Once the server is running, open the generated documentation at: http://localhost:8080/docs


### Using the routes

You will find detailed instructions in the live documentation when your server is up, but here are some examples to use your available API routes:

#### Image classification

Using the following image:
<img src="https://m.media-amazon.com/images/I/517Nh08xqkL._AC_SX425_.jpg" width="50%" height="50%">

with this snippet:

```python
import requests

with open("/path/to/your/img.jpg", "rb") as f:
    print(requests.post("http://localhost:8080/classification", files={"file": f}).json())
```

should yield
```json
{"value": "French horn", "confidence": 0.9685316681861877}
```
