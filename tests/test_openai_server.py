import asyncio
import importlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

openai_server = importlib.import_module("examples.openai_server")


class DummyCloneModel:
    def __init__(self) -> None:
        self.clone_calls = []
        self.clone_stream_calls = []
        self.loaded_speaker_embeddings = []
        self.extract_calls = []

    def generate_voice_clone(self, **kwargs):
        self.clone_calls.append(kwargs)
        return [np.zeros(16, dtype=np.float32)], openai_server.SAMPLE_RATE

    def generate_voice_clone_streaming(self, **kwargs):
        self.clone_stream_calls.append(kwargs)
        yield np.zeros(8, dtype=np.float32), openai_server.SAMPLE_RATE, {"steps": 1}

    def build_voice_clone_prompt_from_embedding(self, speaker_embedding):
        self.loaded_speaker_embeddings.append(speaker_embedding)
        return {
            "ref_code": [None],
            "ref_spk_embedding": [speaker_embedding],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }

    def extract_speaker_embedding(self, ref_audio):
        self.extract_calls.append(ref_audio)
        return torch.ones(1, 4, dtype=torch.bfloat16)


class DummyCustomModel:
    def __init__(self) -> None:
        self.custom_calls = []
        self.custom_stream_calls = []

    def generate_custom_voice(self, **kwargs):
        self.custom_calls.append(kwargs)
        return [np.zeros(16, dtype=np.float32)], openai_server.SAMPLE_RATE

    def generate_custom_voice_streaming(self, **kwargs):
        self.custom_stream_calls.append(kwargs)
        yield np.zeros(8, dtype=np.float32), openai_server.SAMPLE_RATE, {"steps": 1}


async def _read_streaming_response(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


def test_clone_request_instruct_is_passed_to_non_streaming_generation(monkeypatch):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")
    monkeypatch.setattr(openai_server, "_to_mp3_bytes", lambda _pcm, _sr: b"fake-mp3")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "alloy": {
                "ref_audio": "ref.wav",
                "ref_text": "hello",
                "language": "English",
                "instruct": "default clone instruct",
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "alloy")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="alloy",
        response_format="mp3",
        instruct="speak with more urgency",
    )

    response = asyncio.run(openai_server.create_speech(request))

    assert response.media_type == "audio/mpeg"
    assert model.clone_calls[0]["instruct"] == "speak with more urgency"


def test_clone_request_seed_is_passed_to_non_streaming_generation(monkeypatch):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")
    monkeypatch.setattr(openai_server, "_to_mp3_bytes", lambda _pcm, _sr: b"fake-mp3")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "alloy": {
                "ref_audio": "ref.wav",
                "ref_text": "hello",
                "language": "English",
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "alloy")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="alloy",
        response_format="mp3",
        seed=1234,
    )

    response = asyncio.run(openai_server.create_speech(request))

    assert response.media_type == "audio/mpeg"
    assert model.clone_calls[0]["seed"] == 1234


def test_custom_request_instruct_overrides_voice_default_for_streaming(monkeypatch):
    model = DummyCustomModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "custom")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "vivian": {
                "speaker": "vivian",
                "language": "Auto",
                "instruct": "default custom instruct",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "vivian")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="vivian",
        response_format="wav",
        instruct="sound like a calm storyteller",
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.custom_stream_calls[0]["instruct"] == "sound like a calm storyteller"


def test_custom_streaming_uses_voice_default_instruct_when_request_omits_it(monkeypatch):
    model = DummyCustomModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "custom")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "vivian": {
                "speaker": "vivian",
                "language": "Auto",
                "instruct": "default custom instruct",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "vivian")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="vivian",
        response_format="pcm",
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/pcm"
    assert body
    assert model.custom_stream_calls[0]["instruct"] == "default custom instruct"


def test_custom_request_seed_is_passed_to_streaming_generation(monkeypatch):
    model = DummyCustomModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "custom")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "vivian": {
                "speaker": "vivian",
                "language": "Auto",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "vivian")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="vivian",
        response_format="wav",
        seed=4321,
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.custom_stream_calls[0]["seed"] == 4321


def test_custom_request_language_overrides_voice_default(monkeypatch):
    model = DummyCustomModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "custom")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "vivian": {
                "speaker": "vivian",
                "language": "Auto",
                "instruct": "default custom instruct",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "vivian")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="vivian",
        response_format="wav",
        language="Chinese",
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.custom_stream_calls[0]["language"] == "Chinese"


def test_clone_multipart_voice_clone_pt_is_passed_to_streaming_generation(monkeypatch):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "alloy": {
                "language": "English",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "alloy")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    buf = BytesIO()
    torch.save(torch.full((1, 4), 3.0, dtype=torch.bfloat16), buf)

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/speech",
        data={
            "input": "hello world",
            "voice": "alloy",
            "response_format": "wav",
        },
        files={"voice_clone_pt": ("speaker.pt", buf.getvalue(), "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert model.clone_stream_calls[0]["ref_audio"] is None
    assert torch.equal(
        model.clone_stream_calls[0]["voice_clone_prompt"]["ref_spk_embedding"][0].cpu(),
        torch.full((1, 4), 3.0, dtype=torch.bfloat16),
    )


def test_clone_voice_config_supports_static_speaker_pt(monkeypatch, tmp_path):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")
    monkeypatch.setattr(openai_server, "_voice_clone_pt_cache", {})
    speaker_pt = tmp_path / "speaker.pt"
    torch.save(torch.full((1, 4), 5.0, dtype=torch.bfloat16), speaker_pt)
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "alloy": {
                "speaker_pt": str(speaker_pt),
                "language": "English",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "alloy")
    monkeypatch.setattr(openai_server, "SAMPLE_RATE", 24000)

    request = openai_server.SpeechRequest(
        input="hello world",
        voice="alloy",
        response_format="wav",
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.clone_stream_calls[0]["ref_audio"] is None
    assert torch.equal(
        model.clone_stream_calls[0]["voice_clone_prompt"]["ref_spk_embedding"][0].cpu(),
        torch.full((1, 4), 5.0, dtype=torch.bfloat16),
    )


def test_voice_clone_pt_endpoint_returns_serialized_pt(monkeypatch):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/voice-clone/pt",
        data={"filename": "managed-speaker.pt"},
        files={"ref_audio": ("ref.wav", b"fake wav payload", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "managed-speaker.pt" in response.headers["content-disposition"]
    speaker_embedding = torch.load(BytesIO(response.content), map_location="cpu", weights_only=True)
    assert torch.equal(speaker_embedding, torch.ones(1, 4, dtype=torch.bfloat16))
    assert model.extract_calls


def test_custom_mode_rejects_voice_clone_pt(monkeypatch):
    model = DummyCustomModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "custom")
    monkeypatch.setattr(
        openai_server,
        "voices",
        {
            "vivian": {
                "speaker": "vivian",
                "language": "Auto",
                "chunk_size": 8,
            }
        },
    )
    monkeypatch.setattr(openai_server, "default_voice", "vivian")

    with pytest.raises(openai_server.HTTPException) as exc_info:
        asyncio.run(
            openai_server.create_speech(
                openai_server.SpeechRequest(
                    input="hello world",
                    voice="vivian",
                    response_format="wav",
                ),
                request_voice_clone_prompt={
                    "ref_code": [None],
                    "ref_spk_embedding": [torch.ones(1, 4, dtype=torch.bfloat16)],
                    "x_vector_only_mode": [True],
                    "icl_mode": [False],
                },
            )
        )

    assert exc_info.value.status_code == 400
