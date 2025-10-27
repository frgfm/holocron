PY_DIR = .
DEMO_DIR = ./demo
TESTS_DIR = ./tests
DOCS_DIR = ./docs
BACKEND_DIR = ./api
PACKAGE_DIR = ${PY_DIR}/holocron
PYPROJECT_FILE = ${PY_DIR}/pyproject.toml
PYTHON_REQ_FILE = /tmp/requirements.txt

DOCKERFILE_PATH = ./api/Dockerfile
PKG_TEST_DIR = ${PY_DIR}/tests
API_CONFIG_FILE = ${BACKEND_DIR}/pyproject.toml
API_LOCK_FILE = ${BACKEND_DIR}/uv.lock
API_REQ_FILE = ${BACKEND_DIR}/requirements.txt
DEMO_REQ_FILE = ${DEMO_DIR}/requirements.txt
DEMO_SCRIPT = ${DEMO_DIR}/app.py
REPO_OWNER ?= frgfm
REPO_NAME ?= holocron
DOCKER_NAMESPACE ?= ghcr.io/${REPO_OWNER}
DOCKER_TAG ?= latest
DOCKER_PLATFORM ?= linux/amd64
PYTHON_REQ_FILE = /tmp/requirements.txt

.PHONY: help install install-quality lint-check lint-format precommit typing-check deps-check quality style init-gh-labels init-gh-settings install-mintlify start-mintlify

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

########################################################
# Install & Setup
########################################################

venv:
	uv venv --python 3.11

install: ${PY_DIR} ${PYPROJECT_FILE} ## Install the core library
	uv pip install -e ${PY_DIR}

set-version: ${GLOBAL_PYPROJECT} ${BACKEND_PYPROJECT} ## Set the version in the pyproject.toml file
	uv version --frozen --no-build ${BUILD_VERSION}
	uv version --frozen --no-build --project ${BACKEND_DIR} ${BUILD_VERSION}

########################################################
# Code checks
########################################################


install-quality: ${PY_DIR} ${PYPROJECT_FILE} ## Install with quality dependencies
	uv pip install -e '${PY_DIR}[quality]'

lint-check: ${PYPROJECT_FILE} ## Check code formatting and linting
	ruff format --check . --config ${PYPROJECT_FILE}
	ruff check . --config ${PYPROJECT_FILE}

lint-format: ${PYPROJECT_FILE} ## Format code and fix linting issues
	ruff format . --config ${PYPROJECT_FILE}
	ruff check --fix . --config ${PYPROJECT_FILE}

precommit: ${PYPROJECT_FILE} .pre-commit-config.yaml ## Run pre-commit hooks
	pre-commit run --all-files

typing-check: ${PYPROJECT_FILE} ## Check type annotations
	uvx ty check .

deps-check: .github/verify_deps_sync.py ## Check dependency synchronization
	uv run --script .github/verify_deps_sync.py

# this target runs checks on all files
quality: lint-check typing-check deps-check ## Run all quality checks

style: precommit ## Format code and run pre-commit hooks

########################################################
# Builds
########################################################

build: ${PYPROJECT_FILE} ## Build the package
	uv build ${PY_DIR}

publish: ${PY_DIR} ## Publish the package to PyPI
	uv publish --trusted-publishing always

########################################################
# Tests
########################################################

install-test: ${PY_DIR} ${PYPROJECT_FILE} ## Install with test dependencies
	uv pip install -e '${PY_DIR}[test]'

test: ${PYPROJECT_FILE} ## Run the tests
	uv run pytest --cov-report xml


########################################################
# Docs
########################################################

install-docs: ${PYPROJECT_FILE}
	uv pip install -e ".[docs]"

# Build documentation for current version
docs-latest: ${DOCS_DIR}
	uv run sphinx-build ${DOCS_DIR}/source ${DOCS_DIR}/_build -a

# Check that docs can build
docs-full: ${DOCS_DIR}
	cd ${DOCS_DIR} && bash build.sh

########################################################
# Demo
########################################################

install-demo: ${PYPROJECT_FILE}
	uv pip install -e ".[demo]"

# Run the Gradio demo
run-demo: ${DEMO_FILE}
	uv run streamlit run ${DEMO_FILE}

########################################################
# Backend
########################################################

lock-backend: ${BACKEND_DIR} ${BACKEND_PYPROJECT} ## Lock the backend dependencies
	uv lock --project ${BACKEND_DIR}

install-backend: ${BACKEND_DIR} ${BACKEND_PYPROJECT} ## Install the backend deps
	uv export --project ${BACKEND_DIR} --no-hashes --locked --no-dev -o ${PYTHON_REQ_FILE}
	uv pip install -r ${PYTHON_REQ_FILE}

build-backend: ${BACKEND_DIR} ${BACKEND_DOCKERFILE} ## Build the backend container
	docker buildx build --platform ${DOCKER_PLATFORM} -f ${BACKEND_DOCKERFILE} -t ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG} ${BACKEND_DIR}

push-backend: build-backend
	docker push ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG}

########################################################
# Run backend
########################################################

uvicorn-backend: ${BACKEND_DIR} .env
	uv --project ${BACKEND_DIR} run uvicorn app.main:app --reload --reload-dir ${BACKEND_DIR} --host 0.0.0.0 --port 3000 --proxy-headers --use-colors --log-level info --app-dir ${BACKEND_DIR} --env-file .env

start-backend: build-backend ${BACKEND_DIR}
	docker run -p 3000:3000 ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG}

stop-backend: ${BACKEND_DIR}
	docker stop ${DOCKER_NAMESPACE}/${REPO_NAME}-backend:${DOCKER_TAG}

test-backend:  ${API_CONFIG_FILE} ${PYTHON_LOCK_FILE} ${DOCKERFILE_PATH} ${BACKEND_DIR}/tests
	uv export --no-hashes --locked --extra test -q -o ${API_REQ_FILE} --project ${API_DIR}
	docker compose -f ${BACKEND_DIR}/docker-compose.yml up -d --wait --build
	- docker compose -f ${BACKEND_DIR}/docker-compose.yml exec -T backend pytest tests/
	docker compose -f ${BACKEND_DIR}/docker-compose.yml down
