from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openai import OpenAI

from thesis.vlm.vlm_adapter import PreparedRequest, VLMClient, VLMResponse


class OpenAIAdapter(VLMClient):
    provider = "openai"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        pricing_basis: Mapping[str, Any] | None = None,
        client: Any = None,
        cache_dir: Path | None = None,
    ):
        super().__init__(config, pricing_basis=pricing_basis, cache_dir=cache_dir)
        self.client = client if client is not None else OpenAI()

    def _request_settings(self) -> dict[str, Any]:
        parameters = self.config["parameters"]
        return {
            "model": self.config["model"],
            "reasoning": parameters["reasoning"],
            "image": {"mime_type": "image/png", "detail": parameters["detail"]},
        }

    def _request(self, request: PreparedRequest) -> Any:
        parameters = self.config["parameters"]
        return self.client.responses.create(
            model=self.config["model"],
            reasoning=parameters["reasoning"],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{request.image_base64}",
                            "detail": parameters["detail"],
                        },
                    ],
                }
            ],
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
        output_details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = getattr(output_details, "reasoning_tokens", None)
        refusal = None
        for output in getattr(raw_response, "output", []):
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) == "refusal":
                    refusal = getattr(content, "refusal", "refusal")
        if getattr(raw_response, "error", None) is not None:
            refusal = str(raw_response.error)
        elif getattr(raw_response, "status", None) not in (None, "completed"):
            refusal = str(raw_response.status)

        return self._response(
            raw_response=raw_response,
            request=request,
            settings=settings,
            timestamp=timestamp,
            latency=latency,
            raw_text=raw_response.output_text or "",
            returned_model_id=str(raw_response.model),
            provider_request_id=raw_response.id,
            input_tokens=usage.input_tokens,
            image_tokens=self.config.get("image_tokens_512"),
            image_token_count_kind=self.config.get("image_token_count_kind"),
            output_tokens=usage.output_tokens,
            thinking_tokens=thinking_tokens,
            thinking_billed_separately=False,
            refusal_error_state=refusal,
        )
