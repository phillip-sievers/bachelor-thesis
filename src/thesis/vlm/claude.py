import os
from collections.abc import Mapping
from typing import Any

from anthropic import Anthropic

from thesis.vlm.vlm_adapter import PreparedRequest, VLMClient, VLMResponse


class AnthropicAdapter(VLMClient):
    provider = "anthropic"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        pricing_basis: Mapping[str, Any] | None = None,
        client: Any = None,
        workspace_id: str | None = None,
        max_tokens: int = 1024,
    ):
        super().__init__(config, pricing_basis=pricing_basis)
        self.client = client if client is not None else Anthropic()
        self.workspace_id = workspace_id or os.getenv("ANTHROPIC_WORKSPACE_ID")
        self.max_tokens = max_tokens

    def _request_settings(self) -> dict[str, Any]:
        settings = {
            "model": self.config["model"],
            "max_tokens": self.max_tokens,
            **self.config["parameters"],
            "image": {"media_type": "image/png"},
        }
        if self.workspace_id:
            settings["workspace_id"] = self.workspace_id
        return settings

    def _request(self, request: PreparedRequest) -> Any:
        kwargs = {
            "model": self.config["model"],
            "max_tokens": self.max_tokens,
            **self.config["parameters"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": request.image_base64,
                            },
                        },
                        {"type": "text", "text": request.prompt},
                    ],
                }
            ],
        }
        if self.workspace_id:
            kwargs["workspace_id"] = self.workspace_id
        return self.client.messages.create(**kwargs)

    def _parse_response(
        self,
        raw_response: Any,
        request: PreparedRequest,
        settings: dict[str, Any],
        timestamp: float,
        latency: float,
    ) -> VLMResponse:
        raw_text = "".join(block.text for block in raw_response.content if block.type == "text")
        usage = raw_response.usage
        output_details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = getattr(output_details, "thinking_tokens", None)
        refusal = None
        if raw_response.stop_reason == "refusal":
            refusal = str(raw_response.stop_details or "refusal")

        return self._response(
            raw_response=raw_response,
            request=request,
            settings=settings,
            timestamp=timestamp,
            latency=latency,
            raw_text=raw_text,
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
