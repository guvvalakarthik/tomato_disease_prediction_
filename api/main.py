from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import APIError
from .images import decode_image, read_upload, to_model_array
from .inference import ModelRuntime
from .schemas import (
    ClassProbability,
    ErrorBody,
    Explanation,
    HealthResponse,
    ModelIdentity,
    PredictionResponse,
)


LOGGER = logging.getLogger("tomatoguard")


def create_app(
    settings: Settings | None = None, runtime: ModelRuntime | None = None
) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, config.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        app.state.model_error = None
        if runtime is not None:
            app.state.runtime = runtime
        else:
            try:
                app.state.runtime = ModelRuntime(
                    config.model_path, config.metadata_path
                )
                app.state.runtime.load()
            except Exception as exc:
                app.state.runtime = None
                app.state.model_error = str(exc)
                LOGGER.exception("model_load_failed")
        yield

    app = FastAPI(
        title="TomatoGuard API",
        version="1.0.0",
        description="Calibrated tomato leaf disease screening API.",
        lifespan=lifespan,
    )
    app.state.settings = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "request_complete method=%s path=%s status=%s latency_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        body = ErrorBody(
            request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
            code=exc.code,
            message=exc.message,
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/health/ready", response_model=HealthResponse, status_code=200)
    async def ready(request: Request) -> HealthResponse:
        loaded = getattr(request.app.state, "runtime", None)
        if loaded is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="not_ready",
                    detail=request.app.state.model_error or "Model is not loaded.",
                ).model_dump(),
            )
        return HealthResponse(
            status="ok",
            model_version=loaded.metadata.version,
            model_sha256=loaded.sha256,
            class_count=len(loaded.metadata.classes),
        )

    @app.post("/v1/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        explain: bool = False,
    ) -> PredictionResponse:
        loaded = getattr(request.app.state, "runtime", None)
        if loaded is None:
            raise APIError(503, "model_not_ready", "The model is not ready.")

        payload = await read_upload(file, config.max_upload_bytes)
        image = decode_image(payload, config.max_image_pixels)
        batch = to_model_array(image, loaded.metadata.input_size)
        probabilities, heatmap = loaded.predict(batch, include_explanation=explain)

        ranked = probabilities.argsort()[::-1][:3]
        top = [
            ClassProbability(
                class_id=loaded.metadata.classes[index]["id"],
                label=loaded.metadata.classes[index]["label"],
                probability=float(probabilities[index]),
            )
            for index in ranked
        ]
        accepted = top[0].probability >= loaded.metadata.confidence_threshold
        reason = None
        if not accepted:
            reason = (
                "The image is below the validated confidence threshold. "
                "Try another clear tomato-leaf photo or consult an expert."
            )
        explanation = (
            Explanation(method="class_activation_map", png_base64=heatmap)
            if heatmap
            else None
        )
        return PredictionResponse(
            request_id=request.state.request_id,
            status="predicted" if accepted else "uncertain",
            model=ModelIdentity(
                version=loaded.metadata.version, sha256=loaded.sha256
            ),
            prediction=top[0] if accepted else None,
            top_predictions=top,
            uncertainty_reason=reason,
            explanation=explanation,
            disclaimer=loaded.metadata.disclaimer,
        )

    return app


app = create_app()
