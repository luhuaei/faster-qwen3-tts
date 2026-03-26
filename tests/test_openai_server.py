import asyncio
import importlib
import io
import sys
import types
from pathlib import Path

import numpy as np
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

    def generate_voice_clone(self, **kwargs):
        self.clone_calls.append(kwargs)
        return [np.zeros(16, dtype=np.float32)], openai_server.SAMPLE_RATE

    def generate_voice_clone_streaming(self, **kwargs):
        self.clone_stream_calls.append(kwargs)
        yield np.zeros(8, dtype=np.float32), openai_server.SAMPLE_RATE, {"steps": 1}


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


class DummyPromptExtractionModel:
    def __init__(self) -> None:
        self.prompt_calls = []
        self.model = types.SimpleNamespace(create_voice_clone_prompt=self._create_voice_clone_prompt)

    def _create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return [types.SimpleNamespace(ref_spk_embedding=torch.zeros(1, 4, dtype=torch.bfloat16))]


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


def test_clone_request_temperature_and_do_sample_are_passed_to_non_streaming_generation(monkeypatch):
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
        temperature=0.45,
        do_sample=False,
    )

    response = asyncio.run(openai_server.create_speech(request))

    assert response.media_type == "audio/mpeg"
    assert model.clone_calls[0]["temperature"] == 0.45
    assert model.clone_calls[0]["do_sample"] is False


def test_custom_request_temperature_and_do_sample_are_passed_to_streaming_generation(monkeypatch):
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
        temperature=0.55,
        do_sample=False,
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.custom_stream_calls[0]["temperature"] == 0.55
    assert model.custom_stream_calls[0]["do_sample"] is False


def test_custom_request_repetition_penalty_is_passed_to_streaming_generation(monkeypatch):
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
        repetition_penalty=1.15,
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.custom_stream_calls[0]["repetition_penalty"] == 1.15


def test_clone_request_voice_is_case_insensitive(monkeypatch):
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
        voice="ALLOY",
        response_format="mp3",
    )

    response = asyncio.run(openai_server.create_speech(request))

    assert response.media_type == "audio/mpeg"
    assert model.clone_calls[0]["ref_audio"] == "ref.wav"


def test_custom_request_voice_is_case_insensitive(monkeypatch):
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
        voice="ViViAn",
        response_format="wav",
    )

    response = asyncio.run(openai_server.create_speech(request))
    body = asyncio.run(_read_streaming_response(response))

    assert response.media_type == "audio/wav"
    assert body.startswith(b"RIFF")
    assert model.custom_stream_calls[0]["speaker"] == "vivian"


def test_voice_clone_pt_export_endpoint_returns_pt_file(monkeypatch):
    model = DummyPromptExtractionModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/voice-clone/pt",
        files={"ref_audio": ("voice.wav", b"fake-wav-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert 'filename="voice.pt"' in response.headers["content-disposition"]

    speaker_embedding = torch.load(io.BytesIO(response.content), map_location="cpu", weights_only=True)
    assert isinstance(speaker_embedding, torch.Tensor)
    assert tuple(speaker_embedding.shape) == (1, 4)
    assert model.prompt_calls[0]["x_vector_only_mode"] is True
    assert model.prompt_calls[0]["ref_text"] == ""


def test_voice_clone_pt_export_endpoint_rejects_custom_mode(monkeypatch):
    model = DummyPromptExtractionModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "custom")

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/voice-clone/pt",
        files={"ref_audio": ("voice.wav", b"fake-wav-bytes", "audio/wav")},
    )

    assert response.status_code == 400
    assert "clone mode" in response.json()["detail"]


def test_multipart_speech_request_uses_uploaded_voice_clone_pt(monkeypatch):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")
    monkeypatch.setattr(openai_server, "_to_mp3_bytes", lambda _pcm, _sr: b"fake-mp3")
    monkeypatch.setattr(openai_server, "voices", {})
    monkeypatch.setattr(openai_server, "default_voice", None)

    pt_buf = io.BytesIO()
    torch.save(torch.zeros(1, 4, dtype=torch.bfloat16), pt_buf)

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/speech",
        data={"input": "hello world", "response_format": "mp3"},
        files={"voice_clone_pt": ("speaker.pt", pt_buf.getvalue(), "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert "voice_clone_prompt" in model.clone_calls[0]
    assert "ref_audio" not in model.clone_calls[0]
    assert isinstance(model.clone_calls[0]["voice_clone_prompt"]["ref_spk_embedding"][0], torch.Tensor)


def test_multipart_speech_request_rejects_invalid_voice_clone_pt(monkeypatch):
    model = DummyCloneModel()
    monkeypatch.setattr(openai_server, "tts_model", model)
    monkeypatch.setattr(openai_server, "generation_mode", "clone")
    monkeypatch.setattr(openai_server, "voices", {})
    monkeypatch.setattr(openai_server, "default_voice", None)

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/speech",
        data={"input": "hello world", "response_format": "wav"},
        files={"voice_clone_pt": ("speaker.pt", b"not-a-torch-file", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "voice_clone_pt" in response.json()["detail"]


def test_multipart_speech_request_parses_repetition_penalty(monkeypatch):
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

    client = TestClient(openai_server.app)
    response = client.post(
        "/v1/audio/speech",
        data={
            "input": "hello world",
            "voice": "vivian",
            "response_format": "wav",
            "repetition_penalty": "1.2",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert model.custom_stream_calls[0]["repetition_penalty"] == 1.2
