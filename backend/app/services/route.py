"""Road distance between the two ends of a move, from Kakao.

Addresses become coordinates with Kakao Local, then Kakao Mobility's car directions give
the recommended route's distance (m) and duration (s). The distance is what the quote
compares against a mover's base distance (e.g. 10,000 m), so it is never replaced with a
straight-line guess.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import httpx

ATTEMPTS = 2
RETRY_PAUSE_SECONDS = 0.3
# Kakao answers 104 when both points are within 5 m of each other.
SAME_PLACE = 104


@dataclass(frozen=True)
class Point:
    x: float  # longitude
    y: float  # latitude


@dataclass(frozen=True)
class Route:
    distance_m: int
    duration_s: int
    origin: Point
    destination: Point


class RouteLookupFailed(Exception):
    """Kakao could not be reached or answered with an error; worth retrying later."""


class AddressNotLocated(Exception):
    """Kakao has no coordinates for this address; retrying will not help."""

    def __init__(self, role: str):
        super().__init__(f"{role} address not located")
        self.role = role


class Router(Protocol):
    def route(self, origin_address: str, destination_address: str) -> Route: ...


def query_text(address: str) -> str:
    """'서울특별시 중구 퇴계로36길 2 (필동2가)' -> '서울특별시 중구 퇴계로36길 2'."""
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", address)).strip()


class KakaoRouter:
    def __init__(
        self, geocode_url: str, directions_url: str, api_key: str | None,
        timeout: float, budget: float,
    ):
        self.geocode_url = geocode_url
        self.directions_url = directions_url
        self.api_key = api_key
        self.timeout = timeout
        self.budget = budget

    def route(self, origin_address: str, destination_address: str) -> Route:
        if not self.api_key:
            raise RouteLookupFailed("KAKAO_REST_API_KEY is not set")
        deadline = time.monotonic() + self.budget
        with ThreadPoolExecutor(max_workers=2) as pool:
            origin = pool.submit(self._locate, "origin", origin_address, deadline)
            destination = pool.submit(self._locate, "destination", destination_address, deadline)
            start, end = origin.result(), destination.result()
        body = self._get(self.directions_url, {
            "origin": f"{start.x},{start.y}", "destination": f"{end.x},{end.y}",
            "priority": "RECOMMEND", "summary": "true",
        }, deadline)
        try:
            best = body["routes"][0]
            code = int(best["result_code"])
            if code == SAME_PLACE:
                return Route(0, 0, start, end)
            if code != 0:
                raise RouteLookupFailed(f"directions result code {code}")
            summary = best["summary"]
            return Route(int(summary["distance"]), int(summary["duration"]), start, end)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RouteLookupFailed("unexpected directions response") from exc

    def _locate(self, role: str, address: str, deadline: float) -> Point:
        body = self._get(
            self.geocode_url, {"query": query_text(address), "analyze_type": "exact"}, deadline
        )
        try:
            documents = body["documents"]
            if not documents:
                raise AddressNotLocated(role)
            return Point(float(documents[0]["x"]), float(documents[0]["y"]))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RouteLookupFailed("unexpected geocode response") from exc

    def _get(self, url: str, params: dict, deadline: float) -> dict:
        headers = {"Authorization": f"KakaoAK {self.api_key}"}
        last_error = "out of time"
        for attempt in range(ATTEMPTS):
            if attempt:
                if deadline - time.monotonic() <= RETRY_PAUSE_SECONDS:
                    break
                time.sleep(RETRY_PAUSE_SECONDS)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = httpx.get(
                    url, params=params, headers=headers, timeout=min(self.timeout, remaining)
                )
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    continue
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = type(exc).__name__
        raise RouteLookupFailed(f"kakao request failed: {last_error}")
