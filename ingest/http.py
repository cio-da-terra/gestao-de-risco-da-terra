"""Tiny HTTP helper (stdlib only) with retries."""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request

USER_AGENT = "cio-da-terra-ingest/0.1 (AI4Good student project)"


def get(url: str, retries: int = 3, timeout: int = 60, backoff: float = 2.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                # Some servers (IBGE) gzip even when not asked to
                if resp.headers.get("Content-Encoding") == "gzip" or body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                return body
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            # 4xx other than 429 will not get better by retrying
            if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                break
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def get_json(url: str, **kw):
    return json.loads(get(url, **kw).decode("utf-8"))
