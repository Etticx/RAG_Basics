"""Exercise the real Ragas async adapter offline; run inside .eval-venv."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from evaluate_rag import create_metrics


class Response(BaseModel):
    statement: str


class RagasAdapterTests(unittest.TestCase):
    def test_real_adapter_accepts_async_generation(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "offline-test-key",
                                     "RAGAS_DO_NOT_TRACK": "true"}):
            metrics = create_metrics("gemini-3.6-flash", "gemini-embedding-001")
        judge = metrics["faithfulness"].llm
        self.assertTrue(judge.is_async)
        expected = Response(statement="Offline adapter check")
        request = AsyncMock(return_value=expected)
        with patch.object(judge.client.chat.completions, "create", request):
            result = asyncio.run(judge.agenerate("Offline test prompt", Response))
        request.assert_awaited_once()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
