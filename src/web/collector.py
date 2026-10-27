from llm.field_extractor import extract_fields_from_text, extract_fields_from_images
from src.parsers.uos_rss_parser import parse_uos_rss
from src.parsers.contest_korea_parser import parse_contest_korea
from src.pipeline import find_missing_keys

RSS_URL_GENERAL = "https://www.uos.ac.kr/rss/gBoard.do"
RSS_URL_HAKSA = "https://www.uos.ac.kr/rss/hBoard.do"


async def collect_uos():
    parsed = parse_uos_rss(RSS_URL_GENERAL) + parse_uos_rss(RSS_URL_HAKSA)

    extracted_results = []

    for page in parsed:
        page_text = page["title"] + "\n" + page["body"]
        extracted_fields, usage = await extract_fields_from_text(page_text)
        extracted_fields = extracted_fields.to_dict()

        missing_keys = find_missing_keys(extracted_fields)
        if missing_keys == []:
            extracted_results.append(extracted_fields)
            continue

        image_extracted_fields, usage = await extract_fields_from_images(
            page["images"], missing_keys
        )
        image_extracted_fields = image_extracted_fields.to_dict()

        for key, value in image_extracted_fields.items():
            if value is not None:
                extracted_fields[key] = value

        extracted_results.append(extracted_fields)

    return extracted_results
