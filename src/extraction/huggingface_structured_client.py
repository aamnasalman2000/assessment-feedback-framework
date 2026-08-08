from __future__ import annotations

import json
import re
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .semantic_extractors import StructuredLLMClient


class HuggingFaceStructuredClient(
    StructuredLLMClient
):
    """
    Structured-output client backed by a local Hugging Face
    causal language model.

    The model is instructed to return JSON matching the
    supplied schema. The response is then parsed and
    validated later by the pipeline's Pydantic models.
    """

    def __init__(
        self,
        *,
        model: str,
        max_new_tokens: int = 1024,
    ) -> None:
        self._model_name = model
        self.max_new_tokens = max_new_tokens

        print(
            f"Loading tokenizer: {model}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
            )
        )

        print(
            f"Loading model: {model}"
        )

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model,
                quantization_config=(
                    quantization_config
                ),
                device_map="auto",
                trust_remote_code=True,
            )
        )

        self.model.eval()

        print(
            "✓ Hugging Face model loaded"
        )

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
        schema_text = json.dumps(
            output_schema,
            ensure_ascii=False,
            indent=2,
        )

        structured_system_prompt = f"""
{system_prompt}

STRUCTURED OUTPUT REQUIREMENT

Return exactly one valid JSON object.

The object must conform to this JSON schema:

{schema_text}

Important rules:

- Return JSON only.
- Do not return Markdown.
- Do not use code fences.
- Do not include commentary before or after the JSON.
- Use double quotes for all JSON property names and string values.
- Escape line breaks inside JSON strings.
- Escape quotation marks occurring inside JSON string values.
- Do not omit commas between JSON fields.
- Do not add trailing commas.
""".strip()

        messages = [
            {
                "role": "system",
                "content": (
                    structured_system_prompt
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt_text = (
            self._apply_chat_template(
                messages
            )
        )

        model_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(
                self.model.device
            )
            for key, value
            in model_inputs.items()
        }

        input_length = (
            model_inputs[
                "input_ids"
            ].shape[1]
        )

        print(
            f"Prompt token count: {input_length}"
        )

        with torch.inference_mode():
            generated_ids = (
                self.model.generate(
                    **model_inputs,
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=(
                        self.tokenizer
                        .eos_token_id
                    ),
                    eos_token_id=(
                        self.tokenizer
                        .eos_token_id
                    ),
                )
            )

        output_ids = generated_ids[
            0,
            input_length:,
        ]

        content = (
            self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        if not content:
            raise RuntimeError(
                "The model returned an empty "
                "response."
            )

        print(
            "\nRAW MODEL OUTPUT:\n"
        )

        print(content)

        print(
            "\nEND RAW MODEL OUTPUT\n"
        )

        return self._parse_json_response(
            content
        )

    def _apply_chat_template(
        self,
        messages: list[
            dict[str, str]
        ],
    ) -> str:
        """
        Apply the model's native chat template.

        Qwen3 supports enable_thinking=False.
        The fallback allows use with models whose
        templates do not expose that argument.
        """
        try:
            return (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        except TypeError:
            return (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    @staticmethod
    def _strip_markdown_fences(
        content: str,
    ) -> str:
        cleaned = content.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        return cleaned.strip()

    @staticmethod
    def _parse_json_response(
        content: str,
    ) -> dict[str, Any]:
        """
        Parse a model response as JSON.

        Parsing is intentionally conservative.

        Literal control characters are tolerated because
        smaller local models may occasionally emit them
        inside string values. Structural JSON errors are
        not automatically repaired.
        """
        cleaned = (
            HuggingFaceStructuredClient
            ._strip_markdown_fences(
                content
            )
        )

        parse_error: (
            json.JSONDecodeError | None
        ) = None

        try:
            parsed = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:
            parse_error = exc

            try:
                parsed = json.loads(
                    cleaned,
                    strict=False,
                )

            except json.JSONDecodeError:
                parsed = (
                    HuggingFaceStructuredClient
                    ._extract_first_json_object(
                        cleaned
                    )
                )

        if not isinstance(
            parsed,
            dict,
        ):
            raise RuntimeError(
                "The model response must be "
                "a top-level JSON object."
            )

        if parse_error is not None:
            print(
                "JSON required tolerant parsing; "
                f"initial error was: "
                f"{parse_error}"
            )

        return parsed

    @staticmethod
    def _extract_first_json_object(
        content: str,
    ) -> dict[str, Any]:
        """
        Extract the first complete top-level JSON object.

        This helps when a model emits otherwise valid JSON
        with surrounding prose.

        It does not repair missing commas, mismatched
        brackets, malformed quoting, or other structural
        JSON errors.
        """
        start = content.find("{")

        if start < 0:
            raise RuntimeError(
                "The model response did not "
                "contain a JSON object."
            )

        depth = 0
        in_string = False
        escaped = False

        for position in range(
            start,
            len(content),
        ):
            character = content[
                position
            ]

            if in_string:
                if escaped:
                    escaped = False
                    continue

                if character == "\\":
                    escaped = True
                    continue

                if character == '"':
                    in_string = False

                continue

            if character == '"':
                in_string = True
                continue

            if character == "{":
                depth += 1

            elif character == "}":
                depth -= 1

                if depth == 0:
                    candidate = content[
                        start:
                        position + 1
                    ]

                    try:
                        parsed = json.loads(
                            candidate
                        )

                    except json.JSONDecodeError:
                        try:
                            parsed = (
                                json.loads(
                                    candidate,
                                    strict=False,
                                )
                            )

                        except (
                            json.JSONDecodeError
                        ) as exc:
                            raise RuntimeError(
                                "The model returned "
                                "malformed JSON."
                            ) from exc

                    if not isinstance(
                        parsed,
                        dict,
                    ):
                        raise RuntimeError(
                            "The extracted JSON "
                            "value was not an "
                            "object."
                        )

                    return parsed

        raise RuntimeError(
            "The model response contained "
            "an incomplete JSON object."
        )