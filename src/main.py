import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.exceptions import register_exception_handlers
from src.api.middleware import ApiRequestLogMiddleware
from src.api.router import router
from src.collector_job import run_daily_collector
from src.services.extraction import get_models
from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_models()

    load_dotenv()
    x_ai_api_key = settings.X_AI_API_KEY
    collector_task = None

    if x_ai_api_key.strip():
        collector_task = asyncio.create_task(run_daily_collector(x_ai_api_key))
    else:
        logger.warning("X_AI_API_KEY가 없어 정기 수집 작업을 시작하지 않습니다.")

    try:
        yield
    finally:
        if collector_task is not None:
            collector_task.cancel()
            with suppress(asyncio.CancelledError):
                await collector_task


app = FastAPI(lifespan=lifespan)
app.add_middleware(ApiRequestLogMiddleware)
register_exception_handlers(app)
app.include_router(router)
