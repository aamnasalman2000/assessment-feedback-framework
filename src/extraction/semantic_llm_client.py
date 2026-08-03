from __future__ import annotations

import json
from typing import Any

from .semantic_extractors import StructuredLLMClient


class OpenAICompatibleStructuredClient(StructuredLLMClient):
    def __init__(
        self,
        *,
        client: Any,
        model: str,
    ) -> None:
        self.client = client
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_extraction_result",
                    "strict": True,
                    "schema": output_schema,
                },
            },
            temperature=0,
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "The model returned an empty response."
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "The model response could not be decoded as JSON."
            ) from error

        if not isinstance(parsed, dict):
            raise RuntimeError(
                "The model response must be a top-level JSON object."
            )

        return parsed