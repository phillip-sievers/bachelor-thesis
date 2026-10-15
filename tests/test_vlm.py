import base64
import hashlib
import struct
from types import SimpleNamespace

import pytest

from thesis.config import load_config
from thesis.vlm.claude import AnthropicAdapter
from thesis.vlm.google import GoogleAdapter
from thesis.vlm.openai import OpenAIAdapter
from thesis.vlm.run import classify, configured_clients
from thesis.vlm.vlm_adapter import VLMResponse, _cache_key, _prepare_request, parse_binary_answer


class Recorder:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class ShittyRecorder:
    def __init__(self, response, fail_times):
        self.response = response
        self.fail_times = fail_times
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"big time failure {self.calls}")
        return self.response


class RawResponse(SimpleNamespace):
    def model_dump(self, *, mode):
        assert mode == "json"
        return {"id": self.id, "model": self.model}


def png_header(width=512, height=512):
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


@pytest.fixture
def model_config():
    config = load_config("models")
    models = {entry["provider"]: entry for entry in config["models"]}
    return models, config["pricing_basis"]


def cached_openai_adapter(model_config, tmp_path):
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    raw = RawResponse(
        id="resp-cache",
        model="gpt-5.6-sol",
        output_text="Yes",
        usage=usage,
        error=None,
        status="completed",
        output=[],
    )
    requests = Recorder(raw)
    adapter = OpenAIAdapter(
        models["openai"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(responses=requests),
        cache_dir=tmp_path,
    )
    return adapter, requests


def test_cache_key_stable():
    settings = {"model": "gpt-5.6-sol"}
    a = _cache_key(settings, "image-sha", "prompt-sha")
    b = _cache_key(settings, "image-sha", "prompt-sha")
    assert a == b and len(a) == 16


@pytest.mark.parametrize(
    "args",
    [
        ({"model": "other"}, "image-sha", "prompt-sha"),
        ({"model": "gpt-5.6-sol"}, "other-image-sha", "prompt-sha"),
        ({"model": "gpt-5.6-sol"}, "image-sha", "other-prompt-sha"),
    ],
)
def test_cache_key_changes(args):
    base = _cache_key({"model": "gpt-5.6-sol"}, "image-sha", "prompt-sha")
    assert _cache_key(*args) != base


def test_store_then_load(tmp_path, model_config):
    adapter, _ = cached_openai_adapter(model_config, tmp_path)
    response = VLMResponse(
        provider="openai",
        raw_text="Yes",
        parsed_answer="yes",
        requested_model_id="gpt-5.6-sol",
        returned_model_id="gpt-5.6-sol",
        prompt_id="prompt-sha",
        prompt_sha256="prompt-sha",
        image_sha256="image-sha",
        complete_request_settings={"model": "gpt-5.6-sol"},
        provider_request_id="resp-1",
        input_token_count=1,
        image_token_count=308,
        image_token_count_kind="estimate",
        output_token_count=1,
        thinking_token_count=0,
        latency=0.1,
        price_schedule={"tier": "standard"},
        cost=0.000024,
        timestamp=1.0,
        raw_response={"id": "resp-1"},
    )

    adapter.store_atomic(tmp_path, "abc123", response)

    assert adapter.load_cached(tmp_path, "abc123") == response


def test_load_cached_missing_key_returns_none(tmp_path, model_config):
    adapter, _ = cached_openai_adapter(model_config, tmp_path)
    assert adapter.load_cached(tmp_path, "missing") is None


def test_classify_cache_hit_skips_network_call_and_same_response(tmp_path, model_config):
    adapter, requests = cached_openai_adapter(model_config, tmp_path)
    image = png_header()

    first = adapter.classify(image, "prompt")
    second = adapter.classify(image, "prompt")

    assert len(requests.calls) == 1
    assert first.vlm_response == second.vlm_response


def test_openai_adapter_sends_locked_request_and_records_usage(model_config, tmp_path):
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        input_tokens=336,
        output_tokens=6,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    raw = RawResponse(
        id="resp-1",
        model="gpt-5.6-sol-2026-08-01",
        output_text="Yes.",
        usage=usage,
        error=None,
        status="completed",
        output=[],
    )
    requests = Recorder(raw)
    adapter = OpenAIAdapter(
        models["openai"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(responses=requests),
        cache_dir=tmp_path,
    )
    image = png_header()

    classify_respone = adapter.classify(image, "Is the pin on the roof?")
    result = classify_respone.vlm_response

    request = requests.calls[0]
    image_part = request["input"][0]["content"][1]
    assert request["model"] == "gpt-5.6-sol"
    assert request["reasoning"] == {"effort": "low"}
    assert image_part["detail"] == "original"
    assert base64.b64decode(image_part["image_url"].split(",", 1)[1]) == image
    assert result.raw_text == "Yes."
    assert result.requested_model_id == "gpt-5.6-sol"
    assert result.returned_model_id == "gpt-5.6-sol-2026-08-01"
    assert result.provider_request_id == "resp-1"
    assert result.input_token_count == 336
    assert result.image_token_count == 308
    assert result.image_token_count_kind == "estimate"
    assert result.output_token_count == 6
    assert result.thinking_token_count == 0
    assert result.cost == pytest.approx(0.001464)


def test_anthropic_adapter_sends_locked_request_and_records_usage(model_config, tmp_path):
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        input_tokens=399,
        output_tokens=8,
        output_tokens_details=SimpleNamespace(thinking_tokens=3),
    )
    raw = RawResponse(
        id="msg-1",
        model="claude-opus-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Yes."),
        ],
        usage=usage,
        stop_reason="end_turn",
        stop_details=None,
    )
    messages = Recorder(raw)
    adapter = AnthropicAdapter(
        models["anthropic"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(messages=messages),
        cache_dir=tmp_path,
        workspace_id="workspace-1",
    )
    image = png_header()

    classify_respone = adapter.classify(image, "Is the pin on the roof?")
    result = classify_respone.vlm_response

    request = messages.calls[0]
    image_part = request["messages"][0]["content"][0]
    assert request["model"] == "claude-opus-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "low"}
    assert request["workspace_id"] == "workspace-1"
    assert base64.b64decode(image_part["source"]["data"]) == image
    assert result.raw_text == "Yes."
    assert result.image_token_count == 361
    assert result.image_token_count_kind == "exact"
    assert result.thinking_token_count == 3
    assert result.cost == pytest.approx(0.002195)


def test_google_adapter_sends_locked_request_and_records_modality_usage(model_config, tmp_path):
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        total_input_tokens=1112,
        total_output_tokens=1,
        total_thought_tokens=165,
        input_tokens_by_modality=[
            SimpleNamespace(modality="text", tokens=23),
            SimpleNamespace(modality="image", tokens=1089),
        ],
    )
    raw = RawResponse(
        id="interaction-1",
        model="gemini-3.6-flash",
        output_text="Yes",
        usage=usage,
        status="completed",
        errors=None,
    )
    interactions = Recorder(raw)
    adapter = GoogleAdapter(
        models["google"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(interactions=interactions),
        cache_dir=tmp_path,
    )
    image = png_header()

    classify_respone = adapter.classify(image, "Is the pin on the roof?")
    result = classify_respone.vlm_response

    request = interactions.calls[0]
    image_part = request["input"][1]
    assert request["model"] == "gemini-3.6-flash"
    assert request["generation_config"] == {"thinking_level": "low"}
    assert image_part["resolution"] == "high"
    assert base64.b64decode(image_part["data"]) == image
    assert result.image_token_count == 1089
    assert result.image_token_count_kind == "provider_reported"
    assert result.thinking_token_count == 165
    assert result.cost == pytest.approx(0.0014565)


def test_adapter_rejects_non_512_png_before_calling_provider(model_config, tmp_path):
    models, pricing_basis = model_config
    requests = Recorder(None)
    adapter = OpenAIAdapter(
        models["openai"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(responses=requests),
        cache_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="512x512 PNG"):
        adapter.classify(png_header(width=256), "prompt")

    assert requests.calls == []


def test_response_records_content_hashes(model_config, tmp_path):
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    raw = RawResponse(
        id="resp-1",
        model="gpt-5.6-sol",
        output_text="Yes",
        usage=usage,
        error=None,
        status="completed",
        output=[],
    )
    adapter = OpenAIAdapter(
        models["openai"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(responses=Recorder(raw)),
        cache_dir=tmp_path,
    )
    image = png_header()
    prompt = "prompt"

    classify_respone = adapter.classify(image, prompt)
    result = classify_respone.vlm_response

    assert result.image_sha256 == hashlib.sha256(image).hexdigest()
    assert result.prompt_sha256 == hashlib.sha256(prompt.encode()).hexdigest()
    assert result.prompt_id == result.prompt_sha256
    assert result.retry_count == 0
    assert result.parsed_answer == "yes"
    assert result.raw_response == {"id": "resp-1", "model": "gpt-5.6-sol"}


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Yes.", "yes"),
        ("The answer is NO", "no"),
        ("Yes, definitely yes.", "yes"),
        ("yesterday", None),
        ("No—not yes.", None),
        ("I cannot answer.", None),
    ],
)
def test_parse_binary_answer_requires_one_distinct_whole_word(raw_text, expected):
    assert parse_binary_answer(raw_text) == expected


def test_classify_runs_identical_input_through_every_client():
    image = png_header()
    prompt = "same prompt"

    class FakeClient:
        def __init__(self, provider):
            self.provider = provider
            self.calls = []

        def classify(self, passed_image, passed_prompt, task="verification", repeat=0):
            self.calls.append((passed_image, passed_prompt))
            return self.provider

    clients = [FakeClient(provider) for provider in ("openai", "anthropic", "google")]

    results = classify(image, prompt, clients=clients)

    assert results == {provider: provider for provider in ("openai", "anthropic", "google")}
    assert all(client.calls == [(image, prompt)] for client in clients)


def test_configured_clients_creates_resolved_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("thesis.vlm.run.REPO_ROOT", tmp_path)
    stubs = {"openai": object(), "anthropic": object(), "google": object()}

    clients = configured_clients(sdk_clients=stubs)

    by_provider = {client.provider: client for client in clients}
    assert by_provider["openai"].cache_dir == tmp_path / "data/vlm_cache/openai"
    assert by_provider["openai"].cache_dir.exists()


def test_classify_retries_then_succeeds(model_config, tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda x: None)
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        input_tokens=399,
        output_tokens=8,
        output_tokens_details=SimpleNamespace(thinking_tokens=3),
    )
    raw = RawResponse(
        id="msg-1",
        model="claude-opus-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Yes."),
        ],
        usage=usage,
        stop_reason="end_turn",
        stop_details=None,
    )

    messages = ShittyRecorder(raw, fail_times=2)

    adapter = AnthropicAdapter(
        models["anthropic"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(messages=messages),
        cache_dir=tmp_path,
        workspace_id="workspace-1",
    )
    image = png_header()
    prompt = "prompt"

    classify_respone = adapter.classify(image, prompt)
    result = classify_respone.vlm_response

    assert messages.calls == 3
    assert result.raw_text == "Yes."
    assert result.retry_count == 2
    assert classify_respone.cache_hit is False


def test_classify_raises_after_retries(model_config, tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda x: None)
    models, pricing_basis = model_config
    usage = SimpleNamespace(
        input_tokens=399,
        output_tokens=8,
        output_tokens_details=SimpleNamespace(thinking_tokens=3),
    )
    raw = RawResponse(
        id="msg-1",
        model="claude-opus-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Yes."),
        ],
        usage=usage,
        stop_reason="end_turn",
        stop_details=None,
    )

    messages = ShittyRecorder(raw, fail_times=100000)

    adapter = AnthropicAdapter(
        models["anthropic"],
        pricing_basis=pricing_basis,
        client=SimpleNamespace(messages=messages),
        cache_dir=tmp_path,
        workspace_id="workspace-1",
    )
    image = png_header()
    prompt = "prompt"

    with pytest.raises(RuntimeError, match="big time failure 3"):
        adapter.classify(image, prompt)


def test_anthropic_batch_fills_cache_with_live_request_body_and_batch_price(model_config, tmp_path):
    models, pricing_basis = model_config
    usage = SimpleNamespace(input_tokens=399, output_tokens=8, output_tokens_details=SimpleNamespace(thinking_tokens=3))
    message = RawResponse(
        id="msg-1",
        model="claude-opus-5",
        content=[SimpleNamespace(type="text", text="distance: -3 m")],
        usage=usage,
        stop_reason="end_turn",
        stop_details=None,
    )

    class Batches:
        def __init__(self):
            self.requests = []

        def create(self, requests, **_):
            self.requests = requests
            return SimpleNamespace(id="batch-1")

        def retrieve(self, batch_id, **_):
            return SimpleNamespace(processing_status="ended")

        def results(self, batch_id, **_):
            return [
                SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=message))
                for r in self.requests
            ]

    batches = Batches()
    live = Recorder(message)
    live.batches = batches
    adapter = AnthropicAdapter(
        models["anthropic"], pricing_basis=pricing_basis, client=SimpleNamespace(messages=live), cache_dir=tmp_path
    )
    image = png_header()

    assert adapter.batch_classify([(image, "prompt"), (image, "prompt")], task="distance") == ["batch-1"]
    assert len(batches.requests) == 1
    assert batches.requests[0]["params"] == adapter._request_body(_prepare_request(image, "prompt"))
    assert adapter.collect_batch("batch-1") == 1

    result = adapter.classify(image, "prompt", task="distance")
    assert result.cache_hit and live.calls == []
    assert result.vlm_response.parsed_answer == "-3"
    assert result.vlm_response.price_schedule["tier"] == "batch"
    assert result.vlm_response.cost == pytest.approx(0.002195 / 2)
    assert adapter.batch_classify([(image, "prompt")], task="distance") == []
    assert adapter.batch_classify([(image, "other prompt")], task="distance") == ["batch-1"]
    assert adapter.batch_classify([(image, "other prompt")], task="distance") == []  # pending, not resubmitted
