from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.addresses import router as addresses_router
from .api.auth import router as auth_router
from .api.moves import router as moves_router
from .api.routes import router
from .config import Settings
from .services.address_search import JusoAddressSearch
from .services.auth import AuthStore
from .services.building_register import HubBuildingRegister
from .services.media import MediaStorage
from .services.moves import MoveStore


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
        # Same database as users so member moves can reference their account.
        move_store = MoveStore(settings.auth_db_path, settings.guest_move_ttl_seconds)
        move_store.initialize()
        app.state.move_store = move_store
        app.state.media_storage = MediaStorage(settings.media_dir, settings.media_max_bytes)
        key = settings.address_search_api_key
        app.state.address_search = JusoAddressSearch(
            settings.address_search_url,
            key.get_secret_value() if key else None,
            settings.external_api_timeout_seconds,
        )
        data_key = settings.data_go_kr_service_key
        app.state.building_register = HubBuildingRegister(
            settings.building_register_url,
            data_key.get_secret_value() if data_key else None,
            settings.external_api_timeout_seconds,
            settings.building_lookup_budget_seconds,
        )
        yield

    app = FastAPI(title="짐싹 API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type", "Authorization", "X-Guest-Token"],
    )

    app.include_router(router)
    app.include_router(auth_router)
    app.include_router(moves_router)
    app.include_router(addresses_router)
    return app


app = create_app()
