# Copyright (C) 2022-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import time
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from app.config import settings
from app.vision import CLF_CFG, classify_image, decode_image

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    debug=settings.DEBUG,
    version=settings.VERSION,
)


class ClsCandidate(BaseModel):
    """Classification result"""

    value: str = Field(..., json_schema_extra=[{"example": "Wookie"}])
    confidence: float = Field(..., json_schema_extra=[{"gte": 0, "lte": 1}])


# Routing
@app.post("/classification", status_code=status.HTTP_200_OK, summary="Perform image classification")
def classify(file: Annotated[UploadFile, File(...)]) -> ClsCandidate:
    """Runs holocron vision model to analyze the input image"""
    probs = classify_image(decode_image(file.file.read()))

    return ClsCandidate(
        value=CLF_CFG["classes"][probs.argmax()],
        confidence=float(probs.max()),
    )


class Status(BaseModel):
    status: str


# Healthcheck
@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def health_check() -> Status:
    return Status(status="ok")


# Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.PROJECT_DESCRIPTION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
