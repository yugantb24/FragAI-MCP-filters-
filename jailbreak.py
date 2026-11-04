"""Local NeMo-style Snowflake embeddings + NemoGuard ONNX classification.

Uses the official full-precision ONNX embedding export and unnormalized CLS
embeddings, matching NeMo's model_based/models.py pipeline. No remote
inference, pickle deserialization, or downloaded Python code is used.
"""

import logging
import math
import os
from threading import Lock
from typing import Any

from guardrail_utils import InspectionLimitError, decision, message_text

logger = logging.getLogger(__name__)

EMBEDDING_REPO = "Snowflake/snowflake-arctic-embed-m-long"
EMBEDDING_REVISION = "92d97331f1f4b6a366c1f161354b9f3390cc219f"
CLASSIFIER_REPO = "nvidia/NemoGuard-JailbreakDetect"
CLASSIFIER_REVISION = "cc8b97e2bd6c1667c31476eedaa9a75b4d7ed282"
MAX_TOKENS = 2048
TOKEN_OVERLAP = 128
MAX_WINDOWS = 64


class JailbreakFilter:
    """Lazily load local models and reject when any text window is flagged."""

    def __init__(
        self,
        threshold: float = 0.5,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("JAILBREAK_THRESHOLD must be between 0 and 1 (exclusive).")
        self.threshold = threshold
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._tokenizer: Any = None
        self._embedding: Any = None
        self._classifier: Any = None
        # Serialize initialization and CPU inference to bound concurrent memory.
        self._lock = Lock()

    @classmethod
    def from_environment(cls) -> "JailbreakFilter":
        offline = os.getenv("JAILBREAK_LOCAL_FILES_ONLY", "false").lower()
        if offline not in {"true", "false", "1", "0"}:
            raise ValueError("JAILBREAK_LOCAL_FILES_ONLY must be true/false or 1/0.")
        return cls(
            threshold=float(os.getenv("JAILBREAK_THRESHOLD", "0.5")),
            cache_dir=os.getenv("JAILBREAK_MODEL_CACHE") or None,
            local_files_only=offline in {"true", "1"},
        )

    def _load_models(self) -> None:
        if self._classifier is not None:
            return
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession, SessionOptions
        from tokenizers import Tokenizer

        def download(repo: str, revision: str, filename: str) -> str:
            return hf_hub_download(
                repo_id=repo, revision=revision, filename=filename,
                cache_dir=self.cache_dir, local_files_only=self.local_files_only,
            )

        tokenizer = Tokenizer.from_file(download(EMBEDDING_REPO, EMBEDDING_REVISION, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS, stride=TOKEN_OVERLAP)
        tokenizer.no_padding()
        options = SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        embedding = InferenceSession(
            download(EMBEDDING_REPO, EMBEDDING_REVISION, "onnx/model.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"],
        )
        classifier = InferenceSession(
            download(CLASSIFIER_REPO, CLASSIFIER_REVISION, "snowflake.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"],
        )
        self._tokenizer = tokenizer
        self._embedding = embedding
        self._classifier = classifier

    def _window_score(self, encoding: Any) -> float:
        import numpy as np

        values = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
        }
        inputs = {
            item.name: np.asarray([values[item.name]], dtype=np.int64)
            for item in self._embedding.get_inputs()
        }
        # Keep CLS unnormalized: the RF was trained on these embeddings.
        hidden = self._embedding.run(["last_hidden_state"], inputs)[0]
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 768:
            raise ValueError("Unexpected embedding output shape.")
        vector = np.asarray(hidden[:, 0, :], dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite embedding.")
        output = self._classifier.run(None, {"X": vector})
        # NVIDIA's ONNX RF returns labels and a ZipMap of class probabilities.
        score = float(output[1][0][1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid jailbreak probability.")
        return score

    def _detect(self, text: str) -> bool:
        with self._lock:
            self._load_models()
            first = self._tokenizer.encode(text)
            windows = [first, *first.overflowing]
            if len(windows) > MAX_WINDOWS:
                raise InspectionLimitError("Too many token windows.")
            # Inspect overflow instead of silently truncating after 2048 tokens.
            return any(self._window_score(window) >= self.threshold for window in windows)

    def filter_message(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            text = message_text(message)
            if text and self._detect(text):
                return decision(message, accept=False, reason="Blocked: suspected jailbreak attempt.")
            return decision(message, accept=True)
        except InspectionLimitError:
            return decision(message, accept=False, reason="Jailbreak inspection limit exceeded.")
        except Exception as exc:
            logger.error("Jailbreak detection failed (%s)", type(exc).__name__)
            return decision(message, accept=False, reason="Jailbreak detection unavailable; message blocked.")


if __name__ == "__main__":
    # Explicit model warm-up, also useful before enabling an Obot filter.
    detector = JailbreakFilter.from_environment()
    with detector._lock:
        detector._load_models()
    print("Local jailbreak models loaded successfully.")
