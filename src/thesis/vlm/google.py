from collections.abc import Mapping
from pathlib import Path
from typing import Any

from google import genai

from thesis.vlm.vlm_adapter import PreparedRequest, VLMClient, VLMResponse


class GoogleAdapter(VLMClient):
    provider = "google"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        pricing_basis: Mapping[str, Any] | None = None,
        client: Any = None,
        cache_dir: Path | None = None,
    ):
        super().__init__(config, pricing_basis=pricing_basis, cache_dir=cache_dir)
        self.client = client if client is not None else genai.Client()

    def _request_settings(self) -> dict[str, Any]:
        parameters = self.config["parameters"]
        return {
            "model": self.config["model"],
            "generation_config": parameters["thinking_config"],
            "image": {"mime_type": "image/png", "resolution": parameters["media_resolution"]},
        }

    def _request(self, request: PreparedRequest) -> Any:
        parameters = self.config["parameters"]
        return self.client.interactions.create(
            model=self.config["model"],
            input=[
                {"type": "text", "text": request.prompt},
                {
                    "type": "image",
                    "data": request.image_base64,
                    "mime_type": "image/png",
                    "resolution": parameters["media_resolution"],
                },
            ],
            generation_config=parameters["thinking_config"],
        )

    def _parse_response(
        self,
        raw_response: Any,
        request: PreparedRequest,
        settings: dict[str, Any],
        timestamp: float,
        latency: float,
    ) -> VLMResponse:
        usage = raw_response.usage
        image_tokens = None
        for item in usage.input_tokens_by_modality or []:
            if str(item.modality).lower() == "image":
                image_tokens = item.tokens
                break
        image_token_count_kind = "provider_reported" if image_tokens is not None else None
        if image_tokens is None:
            image_tokens = self.config.get("image_tokens_512")
            image_token_count_kind = self.config.get("image_token_count_kind")

        error_state = None
        if raw_response.status != "completed":
            error_state = str(raw_response.errors or raw_response.status)

        return self._response(
            raw_response=raw_response,
            request=request,
            settings=settings,
            timestamp=timestamp,
            latency=latency,
            raw_text=raw_response.output_text or "",
            returned_model_id=str(raw_response.model or self.config["model"]),
            provider_request_id=raw_response.id,
            input_tokens=usage.total_input_tokens,
            image_tokens=image_tokens,
            image_token_count_kind=image_token_count_kind,
            output_tokens=usage.total_output_tokens,
            thinking_tokens=usage.total_thought_tokens,
            thinking_billed_separately=True,
            refusal_error_state=error_state,
        )
