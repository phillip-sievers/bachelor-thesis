import base64
import hashlib
import json
import os
import re
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class VLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    raw_text: str
    parsed_answer: str | None = None
    requested_model_id: str
    returned_model_id: str
    prompt_id: str
    prompt_sha256: str
    image_sha256: str
    complete_request_settings: dict[str, Any]
    provider_request_id: str | None
    input_token_count: int | None
    image_token_count: int | None
    image_token_count_kind: str | None
    output_token_count: int | None
    thinking_token_count: int | None
    latency: float
    price_schedule: dict[str, Any]
    cost: float | None
    timestamp: float
    retry_count: int = 0
    refusal_error_state: str | None = None
    raw_response: dict[str, Any]


class ClassifyResponse(BaseModel):
    vlm_response: VLMResponse
    cache_hit: bool = False


@dataclass(frozen=True)
class PreparedRequest:
    image_base64: str
    image_sha256: str
    prompt: str
    prompt_sha256: str


def parse_binary_answer(raw_text: str) -> str | None:
    matches = set(re.findall(r"\b(?:yes|no)\b", raw_text.casefold()))
    return matches.pop() if len(matches) == 1 else None


def _prepare_request(image: bytes, prompt: str) -> PreparedRequest:
    if len(image) < 24 or image[:8] != b"\x89PNG\r\n\x1a\n" or image[12:16] != b"IHDR":
        raise ValueError("image must be a 512x512 PNG")
    if struct.unpack(">II", image[16:24]) != (512, 512):
        raise ValueError("image must be a 512x512 PNG")

    return PreparedRequest(
        image_base64=base64.b64encode(image).decode("ascii"),
        image_sha256=hashlib.sha256(image).hexdigest(),
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def _serialize_raw_response(raw_response: Any) -> dict[str, Any]:
    if hasattr(raw_response, "model_dump"):
        return raw_response.model_dump(mode="json")
    if isinstance(raw_response, Mapping):
        return dict(raw_response)
    return {"repr": repr(raw_response)}


def _cache_key(settings, image_sha, prompt_sha) -> str:
    raw = f"{settings}_{image_sha}_{prompt_sha}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


class VLMClient(ABC):
    provider: str

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        pricing_basis: Mapping[str, Any] | None = None,
        cache_dir: Path,
    ):
        self.config = deepcopy(dict(config))
        self.pricing_basis = deepcopy(dict(pricing_basis or {}))
        # provider specific cache
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.cache_dir / "log.jsonl"
        self._log_lock = threading.Lock()

    def _append_to_log(self, entry: dict) -> None:
        with self._log_lock:
            with self.log_path.open("a") as file:
                file.write(json.dumps(entry) + "\n")

    def store_atomic(
        self,
        cache_dir: Path,
        key: str,
        parsed_response: VLMResponse,
    ) -> None:
        js = _cache_path(cache_dir, key)

        tmp_js = cache_dir / f"{key}.tmp.json"
        tmp_js.write_text(parsed_response.model_dump_json())
        os.replace(tmp_js, js)

    def load_cached(self, cache_dir: Path, key: str) -> VLMResponse | None:
        js = _cache_path(cache_dir, key)
        if not js.exists():
            return None
        return VLMResponse.model_validate_json(js.read_text())

    def _request_with_backoff(
        self, request: PreparedRequest, cache_key: str, max_attempts: int = 3
    ) -> tuple[Any, float, float, int]:
        for attempt in range(max_attempts):
            try:
                timestamp = time.time()
                started = time.perf_counter()
                raw_response = self._request(request)
                return raw_response, timestamp, time.perf_counter() - started, attempt
            except Exception as e:
                self._append_to_log(
                    {
                        "timestamp": time.time(),
                        "provider": self.provider,
                        "cache_key": cache_key,
                        "cache_hit": False,
                        "error": repr(e),
                    }
                )
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        # pleasing pylance, should be unreachable
        raise RuntimeError(f"_request_with_backoff: max_attempts must be >= 1, got {max_attempts}")

    def classify(self, image: bytes, prompt: str) -> ClassifyResponse:
        request = _prepare_request(image, prompt)
        settings = self._request_settings()
        cache_key = _cache_key(settings, request.image_sha256, request.prompt_sha256)
        cached = self.load_cached(self.cache_dir, cache_key)
        if cached is not None:
            self._append_to_log(
                {
                    "timestamp": time.time(),
                    "provider": self.provider,
                    "cache_key": cache_key,
                    "cache_hit": True,
                }
            )
            return ClassifyResponse(vlm_response=cached, cache_hit=True)

        raw_response, timestamp, latency, attempts = self._request_with_backoff(request, cache_key)
        parsed_response = self._parse_response(raw_response, request, settings, timestamp, latency)
        parsed_response.retry_count = attempts
        self._append_to_log({**parsed_response.model_dump(mode="json"), "cache_hit": False})
        self.store_atomic(self.cache_dir, cache_key, parsed_response)
        return ClassifyResponse(vlm_response=parsed_response, cache_hit=False)

    @abstractmethod
    def _request(self, request: PreparedRequest) -> Any:
        pass

    @abstractmethod
    def _request_settings(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _parse_response(
        self,
        raw_response: Any,
        request: PreparedRequest,
        settings: dict[str, Any],
        timestamp: float,
        latency: float,
    ) -> VLMResponse:
        pass

    def _price_schedule(self) -> dict[str, Any]:
        rates = self.config["pricing_usd_per_million_tokens"]["standard"]
        return {"tier": "standard", **self.pricing_basis, **rates}

    def _cost(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        thinking_tokens: int | None,
        *,
        thinking_billed_separately: bool,
    ) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        rates = self.config["pricing_usd_per_million_tokens"]["standard"]
        billable_output = output_tokens
        if thinking_billed_separately:
            billable_output += thinking_tokens or 0
        return (input_tokens * rates["input"] + billable_output * rates["output"]) / 1_000_000

    def _response(
        self,
        *,
        raw_response: Any,
        request: PreparedRequest,
        settings: dict[str, Any],
        timestamp: float,
        latency: float,
        raw_text: str,
        returned_model_id: str,
        provider_request_id: str | None,
        input_tokens: int | None,
        image_tokens: int | None,
        image_token_count_kind: str | None,
        output_tokens: int | None,
        thinking_tokens: int | None,
        thinking_billed_separately: bool,
        refusal_error_state: str | None = None,
    ) -> VLMResponse:
        parsed_answer = parse_binary_answer(raw_text)
        return VLMResponse(
            provider=self.provider,
            raw_text=raw_text,
            parsed_answer=parsed_answer,
            requested_model_id=self.config["model"],
            returned_model_id=returned_model_id,
            prompt_id=request.prompt_sha256,
            prompt_sha256=request.prompt_sha256,
            image_sha256=request.image_sha256,
            complete_request_settings=settings,
            provider_request_id=provider_request_id,
            input_token_count=input_tokens,
            image_token_count=image_tokens,
            image_token_count_kind=image_token_count_kind,
            output_token_count=output_tokens,
            thinking_token_count=thinking_tokens,
            latency=latency,
            price_schedule=self._price_schedule(),
            cost=self._cost(
                input_tokens,
                output_tokens,
                thinking_tokens,
                thinking_billed_separately=thinking_billed_separately,
            ),
            timestamp=timestamp,
            refusal_error_state=refusal_error_state or ("invalid_output" if parsed_answer is None else None),
            raw_response=_serialize_raw_response(raw_response),
        )
