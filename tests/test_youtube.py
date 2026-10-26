"""Unit tests for YouTube URL validation and downloading module."""

from __future__ import annotations

import pytest

from src.data.youtube import is_youtube_url


def test_is_youtube_url_valid():
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    for url in valid_urls:
        assert is_youtube_url(url), f"Should be valid: {url}"


def test_is_youtube_url_invalid():
    invalid_urls = [
        "",
        "https://vimeo.com/123456",
        "https://google.com",
        "not_a_url",
        "https://example.com/video",
    ]
    for url in invalid_urls:
        assert not is_youtube_url(url), f"Should be invalid: {url}"
