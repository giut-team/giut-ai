import asyncio
from urllib.parse import urlparse

from fastapi import APIRouter

from src.api.dto import CompetitionExtractRequest
from src.parsers.contest_korea_parser import parse_contest_korea
from src.services.collector import extract_contest_korea_page
from src.services.extraction import run_pipeline

router = APIRouter()


@router.post("/api/competitions/extract")
async def extract_competition(request: CompetitionExtractRequest):
    domain = (urlparse(request.url).hostname or "").lower()

    if domain in {"contestkorea.com", "www.contestkorea.com"}:
        page = await asyncio.to_thread(parse_contest_korea, request.url)
        return await extract_contest_korea_page(page)

    extracted_fields, _ = await run_pipeline(request.url)
    return extracted_fields
