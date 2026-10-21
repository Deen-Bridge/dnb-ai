"""Optional acoustic adapter contracts, WAV validation and audio route tests.

No checkpoint download, inference service or heavyweight dependency is needed.
"""

import io
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import phoneme_model
from phoneme_alignment import AlignmentError, router
from phoneme_model import HuggingFacePhonemeRecognizer, ModelUnavailable, get_recognizer
from tests.test_phoneme_alignment import reference


def wav_data(channels=1, sample_rate=16000, sample_width=2, frames=1600, silent=False):
    import wave

    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setframerate(sample_rate)
        target.setsampwidth(sample_width)
        target.writeframes((b"\x00" if silent else b"\x01") * channels * sample_width * frames)
    return output.getvalue()


def recognizer(qiraat=None):
    return HuggingFacePhonemeRecognizer("test/phoneme-model", "a" * 40, qiraat or {"hafs"})


class Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        return Tensor(self.values[item])

    def softmax(self, dim):
        assert dim == -1
        values = np.exp(self.values - self.values.max(axis=-1, keepdims=True))
        return Tensor(values / values.sum(axis=-1, keepdims=True))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Extractor:
    sampling_rate = 16000

    def __call__(self, samples, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        assert samples.dtype == np.float32
        return {"input_values": samples}


class Model:
    config = SimpleNamespace(pad_token_id=0, inputs_to_logits_ratio=320)

    def eval(self):
        self.evaluating = True

    def __call__(self, input_values):
        assert len(input_values) == 1600
        return SimpleNamespace(logits=Tensor(np.array([[[0, 8, 0], [8, 0, 0], [0, 0, 8]]])))


def fake_imports(monkeypatch):
    calls = []

    def loader(obj):
        def load(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return obj

        return SimpleNamespace(from_pretrained=load)

    tokenizer = SimpleNamespace(get_vocab=lambda: {"_": 0, "b": 1, "a": 2})
    packages = {
        "torch": SimpleNamespace(inference_mode=nullcontext),
        "transformers": SimpleNamespace(
            AutoFeatureExtractor=loader(Extractor()),
            AutoTokenizer=loader(tokenizer),
            AutoModelForCTC=loader(Model()),
        ),
    }
    monkeypatch.setattr(phoneme_model.importlib, "import_module", packages.__getitem__)
    return calls


def test_adapter_loads_pinned_model_once_and_returns_real_posteriors(monkeypatch):
    calls = fake_imports(monkeypatch)
    engine = recognizer()
    first = engine.recognize(wav_data(), "hafs")
    second = engine.recognize(wav_data(), "hafs")
    assert first == second
    assert first.labels == ["_", "b", "a"]
    assert first.blank_id == 0
    assert first.frame_shift_ms == 20
    assert first.model_id == "test/phoneme-model@" + "a" * 40
    assert np.argmax(first.probabilities, axis=-1).tolist() == [1, 0, 2]
    assert len(calls) == 3
    assert all(call[1]["revision"] == "a" * 40 and call[1]["trust_remote_code"] is False for call in calls)
    assert calls[1][1]["do_phonemize"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"channels": 2},
        {"sample_rate": 8000},
        {"sample_width": 1},
        {"frames": 1599},
        {"frames": 960001},
        {"silent": True},
    ],
)
def test_rejects_unsupported_audio(change):
    with pytest.raises(AlignmentError):
        HuggingFacePhonemeRecognizer.read_audio(wav_data(**change))


def test_rejects_corrupt_and_truncated_audio():
    for data in [b"not a wave", wav_data()[:-10]]:
        with pytest.raises(AlignmentError):
            HuggingFacePhonemeRecognizer.read_audio(data)


def test_pcm16_normalization():
    samples = HuggingFacePhonemeRecognizer.read_audio(wav_data())
    assert samples.shape == (1600,)
    assert samples.dtype == np.float32
    assert np.all(samples == pytest.approx(257 / 32768))


@pytest.mark.parametrize("revision", ["", "main", "a" * 39, "g" * 40])
def test_requires_pinned_checkpoint(revision):
    with pytest.raises(ModelUnavailable, match="commit SHA"):
        HuggingFacePhonemeRecognizer("test/model", revision, {"hafs"})


def test_unknown_or_undeclared_reading_is_not_silently_treated_as_hafs():
    with pytest.raises(ModelUnavailable, match="supported readings"):
        HuggingFacePhonemeRecognizer("test/model", "a" * 40, {"unknown"})
    with pytest.raises(ModelUnavailable, match="does not support"):
        recognizer().recognize(wav_data(), "warsh")


def test_missing_optional_dependency_is_a_clear_service_error(monkeypatch):
    def missing(_):
        raise ImportError("not installed")

    monkeypatch.setattr(phoneme_model.importlib, "import_module", missing)
    with pytest.raises(ModelUnavailable, match="optional dependencies"):
        recognizer().recognize(wav_data(), "hafs")


def test_invalid_checkpoint_contract_fails_explicitly(monkeypatch):
    fake_imports(monkeypatch)
    engine = recognizer()
    engine._load()
    engine._model.config = SimpleNamespace(pad_token_id=0, inputs_to_logits_ratio=-1)
    with pytest.raises(ModelUnavailable, match="inference failed"):
        engine.recognize(wav_data(), "hafs")


def test_environment_configuration_is_lazy(monkeypatch):
    get_recognizer.cache_clear()
    monkeypatch.delenv("ALIGNMENT_MODEL_ID", raising=False)
    monkeypatch.delenv("ALIGNMENT_MODEL_REVISION", raising=False)
    with pytest.raises(ModelUnavailable, match="ALIGNMENT_MODEL_ID"):
        get_recognizer()
    monkeypatch.setenv("ALIGNMENT_MODEL_ID", "test/arabic-phoneme-model")
    monkeypatch.setenv("ALIGNMENT_MODEL_REVISION", "b" * 40)
    monkeypatch.setenv("ALIGNMENT_MODEL_QIRAAT", "hafs, warsh")
    try:
        engine = get_recognizer()
        assert engine.qiraat == {"hafs", "warsh"}
        assert engine._model is None
    finally:
        get_recognizer.cache_clear()


def test_audio_endpoint_calls_model_and_aligns_output(monkeypatch):
    fake_imports(monkeypatch)
    engine = recognizer()
    monkeypatch.setattr(phoneme_model, "get_recognizer", lambda: engine)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        response = api.post(
            "/quran/alignment/audio",
            data={"reference": reference().model_dump_json()},
            files={"audio": ("recitation.wav", wav_data(), "audio/wav")},
        )
        assert response.status_code == 200
        assert [phone["symbol"] for phone in response.json()["phonemes"]] == ["b", "a"]
        assert response.json()["model_id"] == engine.model_id + "@" + engine.revision
        assert (
            api.post(
                "/quran/alignment/audio",
                data={"reference": reference().model_dump_json()},
                files={"audio": ("bad.wav", b"bad")},
            ).status_code
            == 422
        )
        assert (
            api.post(
                "/quran/alignment/audio",
                data={"reference": "not JSON"},
                files={"audio": ("recitation.wav", wav_data())},
            ).status_code
            == 422
        )


def test_audio_size_limit_and_unconfigured_model(monkeypatch):
    from phoneme_alignment import MAX_AUDIO_BYTES

    def unavailable():
        raise ModelUnavailable("Configure a phoneme model")

    monkeypatch.setattr(phoneme_model, "get_recognizer", unavailable)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as api:
        data = {"reference": reference().model_dump_json()}
        response = api.post(
            "/quran/alignment/audio", data=data, files={"audio": ("huge.wav", b"x" * (MAX_AUDIO_BYTES + 1))}
        )
        assert response.status_code == 413
        response = api.post("/quran/alignment/audio", data=data, files={"audio": ("valid.wav", wav_data())})
        assert response.status_code == 503
        assert response.json()["detail"] == "Configure a phoneme model"
