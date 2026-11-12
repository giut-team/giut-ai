import base64
from io import BytesIO
from unittest.mock import MagicMock

import pytest
import requests
import torch
from PIL import Image

from src.services import extraction
from src.utils import image as image_utils


@pytest.fixture
def image_bytes():
    with BytesIO() as buffer:
        Image.new("RGB", (2, 3), "red").save(buffer, format="PNG")
        return buffer.getvalue()


def test_download_base64_image(image_bytes, monkeypatch):
    get = MagicMock()
    monkeypatch.setattr(image_utils.requests, "get", get)
    url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    image = image_utils.download_image(url)
    assert isinstance(image, Image.Image)
    assert image.format == "PNG"
    assert image.size == (2, 3)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    get.assert_not_called()


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_download_http_image(scheme, image_bytes, monkeypatch):
    response = MagicMock()
    response.content = image_bytes
    get = MagicMock()
    get.return_value.__enter__.return_value = response
    monkeypatch.setattr(image_utils.requests, "get", get)
    url = f"{scheme}://example.com/image.png"

    image = image_utils.download_image(url, timeout=5.0)
    assert isinstance(image, Image.Image)
    assert image.format == "PNG"
    assert image.size == (2, 3)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    get.assert_called_once_with(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5.0
    )
    response.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize(
    "url",
    [
        "data:image/png;base64",
        "data:image/png,hello",
        "data:text/plain;base64,aGVsbG8=",
        "data:image/png;base64,%%%%",
        "data:image/png;base64,aGVsbG8",
        "file:///image.png",
        "https:///image.png",
    ],
)
def test_download_rejects_invalid_source(url):
    with pytest.raises(ValueError):
        image_utils.download_image(url)


def test_download_propagates_http_errors(monkeypatch):
    response = MagicMock()
    response.raise_for_status.side_effect = requests.HTTPError("404")
    get = MagicMock()
    get.return_value.__enter__.return_value = response
    monkeypatch.setattr(image_utils.requests, "get", get)

    with pytest.raises(requests.HTTPError):
        image_utils.download_image("https://example.com/missing.png")


def test_classifier_loads_base64_and_skips_invalid_images(image_bytes, monkeypatch):
    classifier = MagicMock()
    classifier.logits.return_value = torch.tensor([10.0])
    monkeypatch.setattr(extraction, "get_models", lambda: (None, None, classifier))
    valid = {
        "src": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    }
    invalid = {"src": "data:image/png;base64,aGVsbG8="}

    assert extraction.run_image_classifier([invalid, valid]) == [valid]
    images = classifier.logits.call_args.args[0]
    assert len(images) == 1
    assert images[0].size == (2, 3)
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
