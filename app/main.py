from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import attack, meta

def create_app() -> FastAPI:
    app = FastAPI(title="DevNeuron - Fast Gradient Sign Method (FGSM) Adversarial Attacks")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _unhandled_exception_handler(_: Request, __: Exception):
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    app.add_exception_handler(Exception, _unhandled_exception_handler)

    app.include_router(meta.router, tags=["meta"])
    app.include_router(attack.router, tags=["attack"])
    return app

app = create_app()
