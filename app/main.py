from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import attack, meta

import logging, logging.config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(title="DevNeuron - Fast Gradient Sign Method (FGSM) Adversarial Attacks")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://fgsm-demo.mumernisar.dev/", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    async def _unhandled_exception_handler(_: Request, __: Exception):
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    app.add_exception_handler(Exception, _unhandled_exception_handler)

    app.include_router(meta.router, tags=["meta"])
    app.include_router(attack.router, tags=["attack"])
    return app

app = create_app()
