from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import router as auth_router
from .api.routes import router
from .config import Settings
from .services.auth import AuthStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = AuthStore(
            settings.auth_db_path,
            settings.auth_session_ttl_seconds,
            settings.auth_refresh_ttl_seconds,
        )
        store.initialize()
        app.state.auth_store = store
        yield

    app = FastAPI(title="짐싹 API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(router)
    app.include_router(auth_router)
    return app


app = create_app()
