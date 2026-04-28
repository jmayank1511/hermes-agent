"""Tests for the NVIDIA/Riva TTS provider in tools/tts_tool.py."""

import base64
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("NVIDIA_API_KEY", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(key, raising=False)


class TestGenerateNvidiaTts:
    def test_missing_api_key_raises_value_error(self, tmp_path):
        from tools.tts_tool import _generate_nvidia_tts

        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            _generate_nvidia_tts("Hello", str(tmp_path / "test.ogg"), {})

    def test_successful_ogg_generation_uses_riva_payload(self, tmp_path, monkeypatch):
        from tools.tts_tool import (
            DEFAULT_NVIDIA_TTS_LANGUAGE,
            DEFAULT_NVIDIA_TTS_SAMPLE_RATE,
            DEFAULT_NVIDIA_TTS_VOICE,
            NVCF_BASE_URL,
            NVCF_TTS_DEFAULT_FUNCTION_ID,
            _generate_nvidia_tts,
        )

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        audio = b"ogg-opus-audio"
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"audio": base64.b64encode(audio).decode("ascii")}

        with patch("requests.post", return_value=response) as post:
            output_path = str(tmp_path / "test.ogg")
            result = _generate_nvidia_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.ogg").read_bytes() == audio
        post.assert_called_once()
        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        assert url == f"{NVCF_BASE_URL}/v2/nvcf/pexec/functions/{NVCF_TTS_DEFAULT_FUNCTION_ID}"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"] == {
            "text": "Hello world",
            "languageCode": DEFAULT_NVIDIA_TTS_LANGUAGE,
            "encoding": "OGG_OPUS",
            "sampleRateHz": DEFAULT_NVIDIA_TTS_SAMPLE_RATE,
            "voiceName": DEFAULT_NVIDIA_TTS_VOICE,
        }

    def test_config_overrides_riva_payload(self, tmp_path, monkeypatch):
        from tools.tts_tool import _generate_nvidia_tts

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        response = MagicMock(status_code=200)
        response.json.return_value = {"audio": base64.b64encode(b"audio").decode("ascii")}
        config = {
            "nvidia": {
                "function_id": "custom-function",
                "voice": "Magpie-Multilingual.EN-US.Masculine-1",
                "language": "en-US",
                "sample_rate_hz": 44100,
            }
        }

        with patch("requests.post", return_value=response) as post:
            _generate_nvidia_tts("Hi", str(tmp_path / "test.ogg"), config)

        assert post.call_args.args[0].endswith("/custom-function")
        assert post.call_args.kwargs["json"]["voiceName"] == "Magpie-Multilingual.EN-US.Masculine-1"
        assert post.call_args.kwargs["json"]["sampleRateHz"] == 44100


class TestTtsDispatcherNvidia:
    def test_dispatcher_routes_to_nvidia(self, tmp_path, monkeypatch):
        import json

        from tools.tts_tool import text_to_speech_tool

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        with patch("tools.tts_tool._generate_nvidia_tts") as generate, patch(
            "tools.tts_tool._load_tts_config", return_value={"provider": "nvidia"}
        ):
            generate.side_effect = lambda text, path, config: tmp_path.joinpath("out.ogg").write_bytes(b"audio") or path
            output_path = str(tmp_path / "out.ogg")
            result = json.loads(text_to_speech_tool("Hello", output_path=output_path))

        assert result["success"] is True
        assert result["provider"] == "nvidia"
        assert result["voice_compatible"] is True
        generate.assert_called_once()


class TestCheckTtsRequirementsNvidia:
    def test_nvidia_key_returns_true_without_other_providers(self, monkeypatch):
        from tools.tts_tool import check_tts_requirements

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        with patch("tools.tts_tool._import_edge_tts", side_effect=ImportError), patch(
            "tools.tts_tool._import_elevenlabs", side_effect=ImportError
        ), patch("tools.tts_tool._import_openai_client", side_effect=ImportError), patch(
            "tools.tts_tool._import_mistral_client", side_effect=ImportError
        ), patch("tools.tts_tool._check_neutts_available", return_value=False), patch(
            "tools.tts_tool._check_kittentts_available", return_value=False
        ):
            assert check_tts_requirements() is True
