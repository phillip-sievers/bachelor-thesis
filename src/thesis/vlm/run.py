from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from thesis.config import REPO_ROOT, load_config
from thesis.vlm.claude import AnthropicAdapter
from thesis.vlm.google import GoogleAdapter
from thesis.vlm.openai import OpenAIAdapter
from thesis.vlm.vlm_adapter import ClassifyResponse, VLMClient

ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "google": GoogleAdapter,
}

# bottleneck is OpenAI w 500RPM, we stay below that cause not in a rush
POOL_SIZE = 7


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
                cache_dir=REPO_ROOT / model["cache_dir"],
            )
        )
    return clients


def classify(
    image: bytes,
    prompt: str,
    *,
    clients: Sequence[VLMClient] | None = None,
) -> dict[str, ClassifyResponse]:
    results = run_dataset([(image, prompt)], clients=clients)
    return {provider: responses[0] for provider, responses in results.items()}


def run_dataset(
    items: list[tuple[bytes, str]],
    *,
    clients: Sequence[VLMClient] | None = None,
) -> dict[str, list[ClassifyResponse]]:
    active_clients = configured_clients() if clients is None else clients

    def run_provider(client: VLMClient) -> list[ClassifyResponse]:
        with ThreadPoolExecutor(max_workers=POOL_SIZE) as pool:
            return list(pool.map(lambda item: client.classify(*item), items))

    with ThreadPoolExecutor(max_workers=len(active_clients)) as outer:
        futures = {client.provider: outer.submit(run_provider, client) for client in active_clients}
        return {provider: future.result() for provider, future in futures.items()}
