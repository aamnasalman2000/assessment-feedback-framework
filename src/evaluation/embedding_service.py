from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


class EmbeddingService:
    """
    Wrapper around Sentence Transformers for
    deterministic semantic similarity.

    Embeddings are L2-normalised during encoding,
    so cosine similarity can be computed directly
    as a dot product.
    """

    def __init__(
        self,
        *,
        model_name: str = (
            DEFAULT_EMBEDDING_MODEL
        ),
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        self.model_name = model_name
        self.batch_size = batch_size

        print(
            "Loading evaluation embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        print(
            "✓ Evaluation embedding model loaded"
        )

    def encode(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        """
        Encode text into normalised sentence
        embeddings.

        Returns an array with shape:

            (number_of_texts, embedding_dimension)
        """
        text_list = [
            self._normalise_text(text)
            for text in texts
        ]

        if not text_list:
            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def cosine_similarity(
        self,
        left: str,
        right: str,
    ) -> float:
        """
        Compute cosine similarity between two
        strings.

        Because embeddings are already normalised,
        cosine similarity is simply their dot
        product.
        """
        embeddings = self.encode(
            [
                left,
                right,
            ]
        )

        if embeddings.shape[0] != 2:
            raise RuntimeError(
                "Expected exactly two embeddings."
            )

        similarity = float(
            np.dot(
                embeddings[0],
                embeddings[1],
            )
        )

        return self._clip_similarity(
            similarity
        )

    def similarity_matrix(
        self,
        left_texts: Iterable[str],
        right_texts: Iterable[str],
    ) -> np.ndarray:
        """
        Compute a cosine-similarity matrix.

        If there are H human texts and G generated
        texts, the result has shape:

            (H, G)

        Entry [i, j] is the cosine similarity
        between human text i and generated text j.
        """
        left_list = list(
            left_texts
        )
        right_list = list(
            right_texts
        )

        left_embeddings = self.encode(
            left_list
        )
        right_embeddings = self.encode(
            right_list
        )

        if (
            left_embeddings.shape[0] == 0
            or right_embeddings.shape[0] == 0
        ):
            return np.empty(
                (
                    len(left_list),
                    len(right_list),
                ),
                dtype=np.float32,
            )

        similarities = (
            left_embeddings
            @ right_embeddings.T
        )

        return np.clip(
            similarities,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

    @property
    def embedding_dimension(
        self,
    ) -> int:
        dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "The embedding model did not "
                "report an embedding dimension."
            )

        return int(
            dimension
        )

    @staticmethod
    def _normalise_text(
        text: str,
    ) -> str:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Embedding input must be a string."
            )

        normalised = " ".join(
            text.split()
        ).strip()

        if not normalised:
            raise ValueError(
                "Embedding input cannot be empty."
            )

        return normalised

    @staticmethod
    def _clip_similarity(
        value: float,
    ) -> float:
        """
        Normalised embeddings can still produce
        tiny floating-point overshoots around
        [-1, 1].
        """
        return max(
            -1.0,
            min(
                1.0,
                value,
            ),
        )