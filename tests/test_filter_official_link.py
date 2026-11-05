import json

from scripts.url_list import TEST_URLS
from src.services.extraction import _filter_official_url


def test_filter_official_url():
    with open("data/text/test/link_block_test.json", "r") as f:
        link_blocks = (json.load(f))["links"]

        for idx, source_url in enumerate(TEST_URLS["COMPETITION"][:]):
            blocks = [block for block in link_blocks if block["page_id"] == idx]
            official_url = _filter_official_url(blocks, source_url)
            print(idx, ":", official_url)
            assert type(official_url) == str or official_url is None
