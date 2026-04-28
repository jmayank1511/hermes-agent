"""Tests for the NVIDIA/Riva TTS provider in tools/tts_tool.py."""

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
        import riva.client
        from riva.client.proto.riva_audio_pb2 import AudioEncoding

        from tools.tts_tool import (
            DEFAULT_NVIDIA_TTS_LANGUAGE,
            DEFAULT_NVIDIA_TTS_SAMPLE_RATE,
            DEFAULT_NVIDIA_TTS_VOICE,
            NVCF_TTS_DEFAULT_FUNCTION_ID,
            _generate_nvidia_tts,
        )

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        audio = b"ogg-opus-audio"
        auth = MagicMock()
        service = MagicMock()
        service.synthesize.return_value = MagicMock(audio=audio)

        with patch.object(riva.client, "Auth", return_value=auth) as auth_cls, patch.object(
            riva.client, "SpeechSynthesisService", return_value=service
        ) as service_cls:
            output_path = str(tmp_path / "test.ogg")
            result = _generate_nvidia_tts("Hello world", output_path, {})

        assert result == output_path
        assert (tmp_path / "test.ogg").read_bytes() == audio
        auth_cls.assert_called_once()
        auth_kwargs = auth_cls.call_args.kwargs
        assert auth_kwargs["uri"] == "grpc.nvcf.nvidia.com:443"
        assert ["authorization", "Bearer test-key"] in auth_kwargs["metadata_args"]
        assert ["function-id", NVCF_TTS_DEFAULT_FUNCTION_ID] in auth_kwargs["metadata_args"]
        service_cls.assert_called_once_with(auth)
        service.synthesize.assert_called_once_with(
            text="Hello world",
            voice_name=DEFAULT_NVIDIA_TTS_VOICE,
            language_code=DEFAULT_NVIDIA_TTS_LANGUAGE,
            encoding=AudioEncoding.OGGOPUS,
            sample_rate_hz=DEFAULT_NVIDIA_TTS_SAMPLE_RATE,
        )

    def test_wav_generation_wraps_linear_pcm(self, tmp_path, monkeypatch):
        import riva.client
        from riva.client.proto.riva_audio_pb2 import AudioEncoding

        from tools.tts_tool import _generate_nvidia_tts

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        service = MagicMock()
        service.synthesize.return_value = MagicMock(audio=b"\x00\x01\x02\x03")
        config = {
            "nvidia": {
                "function_id": "custom-function",
                "voice": "Magpie-Multilingual.EN-US.Masculine-1",
                "language": "en-US",
                "sample_rate_hz": 44100,
            }
        }

        with patch.object(riva.client, "Auth", return_value=MagicMock()) as auth_cls, patch.object(
            riva.client, "SpeechSynthesisService", return_value=service
        ):
            output_path = str(tmp_path / "test.wav")
            _generate_nvidia_tts("Hi", output_path, config)

        assert ["function-id", "custom-function"] in auth_cls.call_args.kwargs["metadata_args"]
        kwargs = service.synthesize.call_args.kwargs
        assert kwargs["voice_name"] == "Magpie-Multilingual.EN-US.Masculine-1"
        assert kwargs["sample_rate_hz"] == 44100
        assert kwargs["encoding"] == AudioEncoding.LINEAR_PCM
        data = (tmp_path / "test.wav").read_bytes()
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WAVE"


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

    def test_default_nvidia_output_path_uses_wav_extension(self, tmp_path, monkeypatch):
        import json
        from pathlib import Path

        from tools import tts_tool

        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))
        with patch("tools.tts_tool._generate_nvidia_tts") as generate, patch(
            "tools.tts_tool._load_tts_config", return_value={"provider": "nvidia"}
        ):
            generate.side_effect = lambda text, path, config: Path(path).write_bytes(b"RIFF....WAVE") or path
            result = json.loads(tts_tool.text_to_speech_tool("Hello"))

        assert result["success"] is True
        assert result["provider"] == "nvidia"
        assert result["file_path"].endswith(".wav")
        generate.assert_called_once()
        assert generate.call_args.args[1].endswith(".wav")


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
