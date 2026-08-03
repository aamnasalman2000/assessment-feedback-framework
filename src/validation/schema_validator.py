# src/validation/schema_validator.py

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class ArtefactValidationError(Exception):
    pass


def load_schema(schema_path: Path) -> dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as file:
        schema = json.load(file)

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ArtefactValidationError(
            f"Invalid schema '{schema_path.name}': {error.message}"
        ) from error

    return schema


def validate_artefact(
    artefact: dict[str, Any],
    schema_path: Path,
) -> None:
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)

    errors = sorted(
        validator.iter_errors(artefact),
        key=lambda error: list(error.absolute_path),
    )

    if not errors:
        return

    messages: list[str] = []

    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path)
        messages.append(f"{location or '<root>'}: {error.message}")

    raise ArtefactValidationError(
        f"Validation failed for {schema_path.name}:\n"
        + "\n".join(messages)
    )