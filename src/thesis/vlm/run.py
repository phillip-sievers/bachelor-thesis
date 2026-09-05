from collections.abc import Mapping, Sequence
from typing import Any

from thesis.config import load_config
from thesis.vlm.claude import AnthropicAdapter
from thesis.vlm.google import GoogleAdapter
from thesis.vlm.openai import OpenAIAdapter
from thesis.vlm.vlm_adapter import VLMClient, VLMResponse

ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "google": GoogleAdapter,
}


def configured_clients(*, sdk_clients: Mapping[str, Any] | None = None) -> list[VLMClient]:
    config = load_config("models")
    injected = sdk_clients or {}
    clients = []
    for model in config["models"]:
        provider = model["provider"]
        clients.append(
            ADAPTERS[provider](
                model,
                pricing_basis=config["pricing_basis"],
                client=injected.get(provider),
            )
        )
    return clients


def classify(
    image: bytes,
    prompt: str,
    *,
    clients: Sequence[VLMClient] | None = None,
) -> dict[str, VLMResponse]:
    active_clients = configured_clients() if clients is None else clients
    return {client.provider: client.classify(image, prompt) for client in active_clients}
