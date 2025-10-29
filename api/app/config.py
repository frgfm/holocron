# Copyright (C) 2022-2025, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import os
import tomllib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

__all__ = ["settings"]

meta_ = tomllib.load((Path(__file__).parent.parent / "pyproject.toml").open("rb"))


class Settings(BaseSettings):
    # State
    PROJECT_NAME: str = "Holocron API template"
    PROJECT_DESCRIPTION: str = "Template API for Computer Vision"
    VERSION: str = meta_["project"]["version"]
    DEBUG: bool = os.environ.get("DEBUG", "") != "False"
    CLF_HUB_REPO: str = Field(
        os.environ.get("CLF_HUB_REPO", "frgfm/rexnet1_5x"),
        json_schema_extra=[{"min_length": 2, "example": "frgfm/rexnet1_5x"}],
    )


settings = Settings()
