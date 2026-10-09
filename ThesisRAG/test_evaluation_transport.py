import asyncio
import unittest

import httpx

from evaluation_transport import PacedAsyncTransport


class TransportTests(unittest.TestCase):
    def test_retry_honors_google_delay_and_paces_next_request(self):
        now, sleeps, calls = [0], [], []

        async def sleep(delay):
            sleeps.append(delay)
            now[0] += delay

        def handler(request):
            calls.append(now[0])
            if len(calls) == 1:
                return httpx.Response(429, json={"error": {"details": [
                    {"@type": "google.rpc.RetryInfo", "retryDelay": "36s"}]}})
            return httpx.Response(200, json={"ok": True})

        async def run():
            transport = PacedAsyncTransport(transport=httpx.MockTransport(handler),
                                            clock=lambda: now[0], sleep=sleep)
            async with httpx.AsyncClient(transport=transport) as client:
                self.assertEqual((await client.post("https://offline.test", json={})).status_code, 200)
                await client.post("https://offline.test", json={})

        asyncio.run(run())
        self.assertEqual(calls, [0, 37, 50])
        self.assertEqual(sleeps, [37, 13])

    def test_permanent_error_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(400, json={"error": "Bad request"})

        async def run():
            transport = PacedAsyncTransport(transport=httpx.MockTransport(handler))
            async with httpx.AsyncClient(transport=transport) as client:
                self.assertEqual((await client.post("https://offline.test", json={})).status_code, 400)

        asyncio.run(run())
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
