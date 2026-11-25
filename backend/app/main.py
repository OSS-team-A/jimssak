from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.addresses import router as addresses_router
from .api.analysis_worker import router as analysis_worker_router
from .api.auth import router as auth_router
from .api.moves import router as moves_router
from .api.routes import router
from .config import Settings
from .services.address_search import JusoAddressSearch
from .services.analysis import AnalysisStore
from .services.auth import AuthStore
from .services.building_register import HubBuildingRegister
from .services.calendar import KasiCalendar
from .services.media import MediaStorage
from .services.moves import MoveStore
from .services.route import KakaoRouter


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
        analysis_store = AnalysisStore(
            settings.auth_db_path,
            settings.analysis_timeout_seconds,
            settings.analysis_demo_seconds,
        )
        analysis_store.initialize()
        app.state.analysis_store = analysis_store
        app.state.analysis_demo = settings.analysis_demo
        worker_key = settings.analysis_worker_key
        app.state.analysis_worker_key = worker_key.get_secret_value() if worker_key else None
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
            settings.building_register_request_timeout_seconds,
            settings.building_lookup_budget_seconds,
        )
        kakao_key = settings.kakao_rest_api_key
        app.state.router = KakaoRouter(
            settings.kakao_geocode_url,
            settings.kakao_directions_url,
            kakao_key.get_secret_value() if kakao_key else None,
            settings.external_api_timeout_seconds,
            settings.route_lookup_budget_seconds,
        )
        app.state.calendar = KasiCalendar(
            settings.kasi_holiday_url,
            settings.kasi_lunar_url,
            data_key.get_secret_value() if data_key else None,
            settings.calendar_request_timeout_seconds,
            settings.calendar_lookup_budget_seconds,
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
    app.include_router(analysis_worker_router)
    return app


app = create_app()
