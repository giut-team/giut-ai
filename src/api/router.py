import asyncio
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import APIRouter
import requests

from src.api.dto import CompetitionExtractRequest, CompetitionExtractResponse
from src.parsers.contest_korea_parser import parse_contest_korea
from src.parsers.html_parser import InaccessibleURLError
from src.parsers.uos_html_parser import parse_uos_html
from src.services.collector import extract_contest_korea_page, extract_uos_page
from src.services.extraction import run_pipeline

router = APIRouter()


async def _fetch_page(parse: Callable[[str], Any], url: str) -> Any:
    """사이트 전용 파서로 페이지를 가져오되, 접속 실패는 InaccessibleURLError로 변환"""
    try:
        return await asyncio.to_thread(parse, url)
    except requests.RequestException as exc:
        raise InaccessibleURLError(str(exc)) from exc


@router.post("/api/competitions/extract", response_model=CompetitionExtractResponse)
async def extract_competition(request: CompetitionExtractRequest):
    domain = (urlparse(request.url).hostname or "").lower()

    if domain in {"uos.ac.kr", "www.uos.ac.kr"}:
        page = await _fetch_page(parse_uos_html, request.url)
        return await extract_uos_page(page)

    if domain in {"contestkorea.com", "www.contestkorea.com"}:
        page = await _fetch_page(parse_contest_korea, request.url)
        return await extract_contest_korea_page(page)

    return await run_pipeline(request.url)
