import base64
import hashlib
import re
import struct
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
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


class VLMClient(ABC):
    provider: str

    def __init__(self, config: Mapping[str, Any], *, pricing_basis: Mapping[str, Any] | None = None):
        self.config = deepcopy(dict(config))
        self.pricing_basis = deepcopy(dict(pricing_basis or {}))

    def classify(self, image: bytes, prompt: str) -> VLMResponse:
        request = _prepare_request(image, prompt)
        settings = self._request_settings()
        timestamp = time.time()
        started = time.perf_counter()
        raw_response = self._request(request)
        latency = time.perf_counter() - started
        return self._parse_response(raw_response, request, settings, timestamp, latency)

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
