import os
from collections.abc import Mapping
from pathlib import Path
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
        cache_dir: Path | None = None,
        workspace_id: str | None = None,
    ):
        super().__init__(config, pricing_basis=pricing_basis, cache_dir=cache_dir)
        self.client = client if client is not None else Anthropic()
        self.workspace_id = workspace_id or os.getenv("ANTHROPIC_WORKSPACE_ID")
        # Anthropic must have max_tokens. 16000 is sufficient for thinking and below the SDK limit without streaming.
        self.max_tokens = self.inference["max_output_tokens"] or 16000

    def _request_settings(self) -> dict[str, Any]:
        settings = {
            "model": self.config["model"],
            "max_tokens": self.max_tokens,
            **self._common_settings({"temperature": "temperature"}),
            **self.config["parameters"],
            "image": {"media_type": "image/png"},
        }
        if self.workspace_id:
            settings["workspace_id"] = self.workspace_id
        return settings

    def _request_body(self, request: PreparedRequest) -> dict[str, Any]:
        return {
            "model": self.config["model"],
            "max_tokens": self.max_tokens,
            **self._common_settings({"temperature": "temperature"}),
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

    def _workspace(self) -> dict[str, Any]:
        return {"workspace_id": self.workspace_id} if self.workspace_id else {}

    def _request(self, request: PreparedRequest) -> Any:
        return self.client.messages.create(**self._request_body(request), **self._workspace())

    batch_max_bytes = 240_000_000  # Limit: 256 MB for each batch.

    def _submit_batch(self, bodies: dict[str, dict[str, Any]]) -> str:
        requests = [{"custom_id": key, "params": body} for key, body in bodies.items()]
        return self.client.messages.batches.create(requests=requests, **self._workspace()).id

    def _batch_results(self, batch_id: str) -> dict[str, Any] | None:
        if self.client.messages.batches.retrieve(batch_id, **self._workspace()).processing_status != "ended":
            return None
        return {
            entry.custom_id: entry.result.message if entry.result.type == "succeeded" else None
            for entry in self.client.messages.batches.results(batch_id, **self._workspace())
        }

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
