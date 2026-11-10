from enum import Enum


class UrlType(Enum):
    SOURCE_ORIGINAL = "SOURCE_ORIGINAL"
    OFFICIAL = "OFFICIAL"


SNS_DOMAINS = {
    "instagram.com",
    "facebook.com",
    "fb.com",
    "x.com",
    "twitter.com",
    "blog.naver.com",
    "band.us",
    "youtube.com",
    "youtu.be",
    "pinterest.com",
    "pin.it",
    "kakao.com",
    "tistory.com",
    "google.com",
    r"^naver.com",
    "naver.me",
    "openstreetmap.org",  # 이벤터스
    "stibee.com",  # 이벤터스
    "catchsecu.com",  # 이벤터스
}

APPLY_PATH_KEYWORDS = {
    "form",
    "forms",
    "apply",
    "application",
    "register",
    "registration",
}

OFFICIAL_PAGE_KEYWORDS = {
    "홈페이지",
    "주최사",
    "공고",
    "바로가기",
    "공식",
    "모집",
    "상세",
    "자세",
}

IRRELEVANT_KEYWORDS = {
    "지원",
    "신청",
    "접수",
    "첨부",
    "파일",
    r"오시는\s길",
    *SNS_DOMAINS,
    *APPLY_PATH_KEYWORDS,
}

UOS_HOSTNAME = "uos.ac.kr"  # 학교 공지
CAMPUSPICK_HOSTNAME = "campuspick.com"  # 캠퍼스픽
DACON_HOSTNAME = "dacon.io"  # 데이콘
DAKER_HOSTNAME = "daker.ai"  # 데이커
