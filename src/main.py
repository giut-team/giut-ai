import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from dotenv import load_dotenv
from fastapi import FastAPI

from src.api.router import router
from src.collector_job import run_daily_collector

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    bearer_token = os.getenv("GIUT_BEARER_TOKEN", "")
    collector_task = None

    if bearer_token.strip():
        collector_task = asyncio.create_task(run_daily_collector(bearer_token))
    else:
        logger.warning("GIUT_BEARER_TOKEN이 없어 정기 수집 작업을 시작하지 않습니다.")

    try:
        yield
    finally:
        if collector_task is not None:
            collector_task.cancel()
            with suppress(asyncio.CancelledError):
                await collector_task


app = FastAPI(lifespan=lifespan)
app.include_router(router)
