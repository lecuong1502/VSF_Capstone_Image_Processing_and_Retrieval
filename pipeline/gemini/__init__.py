"""Shared Gemini client: structured output, content-hash cache, Batch API and a hard spend cap.

Every pipeline step that calls Gemini must go through `GeminiClient` (see CLAUDE.md).
"""

from pipeline.gemini.client import (
    BudgetExceededError,
    GeminiClient,
    GeminiRequest,
    GeminiResult,
    Image,
    Pricing,
    RawResponse,
    Transport,
)
from pipeline.gemini.ledger import UsageLedger
from pipeline.gemini.settings import GeminiSettings

__all__ = [
    "BudgetExceededError",
    "GeminiClient",
    "GeminiRequest",
    "GeminiResult",
    "GeminiSettings",
    "Image",
    "Pricing",
    "RawResponse",
    "Transport",
    "UsageLedger",
]
