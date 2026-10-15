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

from thesis.config import load_config


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


# The last line must be "distance: <integer>", with an optional unit m, meter(s) or metre(s).
_DISTANCE_LINE = re.compile(r"distance:\s*(-?\d+)\s*(?:m|meters?|metres?)?\.?")


def parse_distance_answer(raw_text: str) -> str | None:
    lines = [line.strip() for line in raw_text.casefold().splitlines() if line.strip()]
    match = _DISTANCE_LINE.fullmatch(lines[-1]) if lines else None
    return str(int(match.group(1))) if match else None


PARSERS = {"verification": parse_binary_answer, "distance": parse_distance_answer}


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


def _cache_key(settings, image_sha, prompt_sha, repeat: int = 0) -> str:
    # Repeat 0 keeps the old key. Thus the existing cache entries stay valid.
    raw = f"{settings}_{image_sha}_{prompt_sha}" + (f"_repeat{repeat}" if repeat else "")
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
        self.inference = load_config("inference")
        self.tier = "standard"
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

    def classify(self, image: bytes, prompt: str, task: str = "verification", repeat: int = 0) -> ClassifyResponse:
        request = _prepare_request(image, prompt)
        settings = self._request_settings()
        cache_key = _cache_key(settings, request.image_sha256, request.prompt_sha256, repeat)
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
        parsed_response = self._parsed(raw_response, request, settings, timestamp, latency, task)
        parsed_response.retry_count = attempts
        self._append_to_log({**parsed_response.model_dump(mode="json"), "cache_hit": False})
        self.store_atomic(self.cache_dir, cache_key, parsed_response)
        return ClassifyResponse(vlm_response=parsed_response, cache_hit=False)

    def _parsed(self, raw_response, request, settings, timestamp, latency, task) -> VLMResponse:
        response = self._parse_response(raw_response, request, settings, timestamp, latency)
        response.parsed_answer = PARSERS[task](response.raw_text)
        if response.refusal_error_state is None and response.parsed_answer is None:
            response.refusal_error_state = "invalid_output"
        return response

    # A batch uses the same request bodies as classify. The results go into the cache.
    batch_max_bytes: int | None = None

    def batch_classify(self, items: list[tuple[bytes, str]], task: str = "verification", repeat: int = 0) -> list[str]:
        settings = self._request_settings()
        batch_dir = self.cache_dir / "batches"
        batch_dir.mkdir(exist_ok=True)
        # Do not send a key again if a batch has it. A failed batch request goes live in the next classify.
        seen = {key for path in batch_dir.glob("*.json") for key in json.loads(path.read_text())["items"]}
        batch_ids, chunk, size = [], {}, 0

        def submit():
            batch_id = self._submit_batch({key: body for key, (_, body) in chunk.items()})
            items_meta = {key: [r.image_sha256, r.prompt_sha256] for key, (r, _) in chunk.items()}
            (batch_dir / f"{batch_id}.json").write_text(json.dumps({"task": task, "items": items_meta}))
            batch_ids.append(batch_id)

        for image, prompt in items:
            request = _prepare_request(image, prompt)
            key = _cache_key(settings, request.image_sha256, request.prompt_sha256, repeat)
            if key in seen or _cache_path(self.cache_dir, key).exists():
                continue
            seen.add(key)
            body = self._request_body(request)
            body_size = len(json.dumps(body))
            if chunk and size + body_size > self.batch_max_bytes:
                submit()
                chunk, size = {}, 0
            chunk[key] = (request, body)
            size += body_size
        if chunk:
            submit()
        return batch_ids

    def collect_batch(self, batch_id: str) -> int | None:
        results = self._batch_results(batch_id)
        if results is None:
            return None
        meta = json.loads((self.cache_dir / "batches" / f"{batch_id}.json").read_text())
        settings = self._request_settings()
        self.tier = "batch"
        try:
            stored = 0
            for key, raw_response in results.items():
                if raw_response is None:
                    continue
                image_sha, prompt_sha = meta["items"][key]
                request = PreparedRequest("", image_sha, "", prompt_sha)
                # A batch has no latency value.
                response = self._parsed(raw_response, request, settings, time.time(), 0.0, meta["task"])
                self.store_atomic(self.cache_dir, key, response)
                stored += 1
        finally:
            self.tier = "standard"
        return stored

    def _request_body(self, request: PreparedRequest) -> dict[str, Any]:
        raise NotImplementedError(f"{self.provider}: no batch support")

    def _submit_batch(self, bodies: dict[str, dict[str, Any]]) -> str:
        raise NotImplementedError(f"{self.provider}: no batch support")

    def _batch_results(self, batch_id: str) -> dict[str, Any] | None:
        raise NotImplementedError(f"{self.provider}: no batch support")

    def _common_settings(self, names: Mapping[str, str]) -> dict[str, Any]:
        return {names[key]: value for key, value in self.inference.items() if value is not None and key in names}

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
        rates = self.config["pricing_usd_per_million_tokens"][self.tier]
        return {"tier": self.tier, **self.pricing_basis, **rates}

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
        rates = self.config["pricing_usd_per_million_tokens"][self.tier]
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
        # classify sets parsed_answer and invalid_output for each task.
        return VLMResponse(
            provider=self.provider,
            raw_text=raw_text,
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
            refusal_error_state=refusal_error_state,
            raw_response=_serialize_raw_response(raw_response),
        )
