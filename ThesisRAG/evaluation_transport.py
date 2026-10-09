"""Pace evaluation requests and honor Gemini's transient retry delays."""

import asyncio
import time

import httpx


def retry_delay(response, attempt):
    try:
        return max(0, float(response.headers["retry-after"])) + 1
    except (KeyError, ValueError):
        pass
    try:
        body = response.json()
        if isinstance(body, list):
            body = body[0]
        for detail in body.get("error", {}).get("details", []):
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                return max(0, float(detail["retryDelay"].rstrip("s"))) + 1
    except (ValueError, KeyError, IndexError, AttributeError, TypeError):
        pass
    return 61 if response.status_code == 429 else 15 * (2 ** attempt)


class PacedAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, interval=13, retries=2, transport=None,
                 clock=time.monotonic, sleep=asyncio.sleep):
        self.transport = transport or httpx.AsyncHTTPTransport()
        self.interval = interval
        self.retries = retries
        self.clock, self.sleep = clock, sleep
        self.next_request = 0
        self.lock = asyncio.Lock()

    async def handle_async_request(self, request):
        async with self.lock:
            for attempt in range(self.retries + 1):
                delay = self.next_request - self.clock()
                if delay > 0:
                    await self.sleep(delay)
                self.next_request = self.clock() + self.interval
                response = await self.transport.handle_async_request(request)
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    return response
                await response.aread()
                delay = retry_delay(response, attempt)
                await response.aclose()
                self.next_request = max(self.next_request, self.clock() + delay)
                print(f"Gemini returned HTTP {response.status_code}; retrying after {delay:.0f}s.", flush=True)

    async def aclose(self):
        await self.transport.aclose()
