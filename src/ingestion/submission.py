from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any


IGNORED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}


ARTIFACT_TYPE_BY_EXTENSION = {
    ".lean": "lean_source",
    ".pl": "prolog_source",
    ".owl": "owl_ontology",
    ".owx": "owl_ontology",
    ".rdf": "owl_ontology",
    ".xml": "owl_ontology",
    ".pdf": "document",
    ".txt": "text_document",
    ".md": "text_document",
    ".docx": "document",
}


def discover_submission_files(
    student_directory: Path,
) -> list[Path]:
    if not student_directory.exists():
        raise FileNotFoundError(
            f"Student directory not found: {student_directory}"
        )

    files = [
        path
        for path in student_directory.rglob("*")
        if path.is_file() and path.name not in IGNORED_NAMES
    ]

    return sorted(files, key=lambda path: str(path).lower())


def infer_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()

    return ARTIFACT_TYPE_BY_EXTENSION.get(
        suffix,
        "unknown_artifact",
    )


def read_raw_content(path: Path) -> tuple[str, str]:
    try:
        return path.read_text(encoding="utf-8"), "utf-8"
    except UnicodeDecodeError:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        ), "utf-8"


def build_file_record(
    path: Path,
    student_directory: Path,
    index: int,
) -> dict[str, Any]:
    media_type, _ = mimetypes.guess_type(path.name)
    raw_content, content_encoding = read_raw_content(path)

    suffix = path.suffix.lower().lstrip(".")
    file_format = suffix or "unknown"

    return {
        "artifact_id": f"artifact_{index:03d}",
        "artifact_type": infer_artifact_type(path),
        "source": {
            "filename": path.name,
            "format": file_format,
            "media_type": media_type,
            "path": path.relative_to(student_directory).as_posix(),
        },
        "content": {
            "raw_content": raw_content,
            "content_encoding": content_encoding,
        },
    }


def build_submission(
    *,
    assessment_id: str,
    student_id: str,
    student_directory: Path,
    schema_version: str = "1.0",
) -> dict[str, Any]:
    files = discover_submission_files(student_directory)

    if not files:
        raise ValueError(
            f"No submission files found in {student_directory}"
        )

    artifacts = [
        build_file_record(
            path=path,
            student_directory=student_directory,
            index=index,
        )
        for index, path in enumerate(files, start=1)
    ]

    return {
        "schema_version": schema_version,
        "submission_id": f"{assessment_id}_{student_id}",
        "student_id": student_id,
        "assessment_id": assessment_id,
        "artifacts": artifacts,
    }