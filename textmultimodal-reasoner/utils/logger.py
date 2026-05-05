"""Hugging Face-style conversation logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_hf_example(
    output_path: str | Path,
    *,
    question: str,
    reasoning: str,
    answer: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a JSONL record suitable for later HF dataset ingestion.

    Stored in `.txt` as requested, one JSON object per line.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": datetime.now(timezone.utc).isoformat(),
        "instruction": question,
        "input": "",
        "reasoning": reasoning,
        "output": answer,
        "source": "textmultimodal-reasoner",
        "metadata": metadata or {},
    }

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
