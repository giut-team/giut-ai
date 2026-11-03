import asyncio
from urllib.parse import urlparse

from fastapi import APIRouter

from src.api.dto import CompetitionExtractRequest, CompetitionExtractResponse
from src.parsers.contest_korea_parser import parse_contest_korea
from src.parsers.uos_html_parser import parse_uos_html
from src.services.collector import extract_contest_korea_page, extract_uos_page
from src.services.extraction import run_pipeline

router = APIRouter()


@router.post("/api/competitions/extract", response_model=CompetitionExtractResponse)
async def extract_competition(request: CompetitionExtractRequest):
    domain = (urlparse(request.url).hostname or "").lower()

    if domain in {"uos.ac.kr", "www.uos.ac.kr"}:
        page = await asyncio.to_thread(parse_uos_html, request.url)
        return await extract_uos_page(page)

    if domain in {"contestkorea.com", "www.contestkorea.com"}:
        page = await asyncio.to_thread(parse_contest_korea, request.url)
        return await extract_contest_korea_page(page)

    result, _ = await run_pipeline(request.url)

    return result
