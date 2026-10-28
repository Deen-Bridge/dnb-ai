"""Offline CTC path correctness, streaming, boundary, validation and API tests."""

import itertools
import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from phoneme_alignment import (
    AcousticConfig,
    AcousticFrames,
    AlignmentError,
    AlignmentReference,
    CTCAligner,
    FrameBatch,
    Phone,
    ReferenceVerse,
    ReferenceWord,
    align,
    alignment_stream,
    router,
)


def reference(symbols=("b", "a"), qiraah="hafs") -> AlignmentReference:
    return AlignmentReference(
        qiraah=qiraah,
        reciter="test reciter",
        verses=[
            ReferenceVerse(
                surah=1, ayah=1, words=[ReferenceWord(text="بَ", phonemes=[Phone(symbol=s) for s in symbols])]
            )
        ],
    )


def acoustics(path, labels=("_", "b", "a"), **kwargs) -> AcousticFrames:
    return AcousticFrames(
        labels=list(labels),
        blank_id=0,
        frame_shift_ms=20,
        model_id="synthetic acoustic fixture",
        probabilities=[[float(i == token) for i in range(len(labels))] for token in path],
        **kwargs,
    )


def client():
    app = FastAPI()
    app.include_router(router)
    app.add_api_websocket_route("/stream", alignment_stream)
    return TestClient(app)


def test_variable_duration_timing_and_silence_are_acoustic():
    result = align(reference(), acoustics([0, 0, 1, 1, 1, 0, 2, 2, 0], offset_ms=1000))
    assert [(p.start_ms, p.end_ms) for p in result.phonemes] == [(1040, 1100), (1120, 1160)]
    assert result.words[0].start_ms == result.verses[0].start_ms == 1040
    assert result.words[0].end_ms == result.verses[0].end_ms == 1160
    assert result.frames_processed == 9
    assert result.path_log_probability == 0
    assert result.confidence == 1
    assert result.final


def test_repeated_phonemes_need_a_ctc_blank():
    result = align(reference(("b", "b")), acoustics([1, 1, 0, 1]))
    assert len(result.phonemes) == 2
    assert [(p.start_ms, p.end_ms) for p in result.phonemes] == [(0, 40), (60, 80)]
    with pytest.raises(AlignmentError, match="No valid CTC path"):
        align(reference(("b", "b")), acoustics([1, 1]))


def test_multiple_word_and_verse_boundaries_and_tajweed_metadata():
    target = AlignmentReference(
        qiraah="warsh",
        verses=[
            ReferenceVerse(
                surah=1,
                ayah=1,
                words=[
                    ReferenceWord(text="ب", phonemes=[Phone(symbol="b", tajweed=["qalqalah"])]),
                    ReferenceWord(text="ا", phonemes=[Phone(symbol="a", tajweed=["madd"])]),
                ],
            ),
            ReferenceVerse(
                surah=1,
                ayah=2,
                words=[ReferenceWord(text="ب", phonemes=[Phone(symbol="b", tajweed=["shadda", "ghunnah"])])],
            ),
        ],
    )
    result = align(target, acoustics([0, 1, 1, 0, 2, 2, 0, 1, 0]))
    assert [(w.verse_index, w.word_index) for w in result.words] == [(0, 0), (0, 1), (1, 0)]
    assert [(v.surah, v.ayah, v.start_ms, v.end_ms) for v in result.verses] == [(1, 1, 20, 120), (1, 2, 140, 160)]
    assert result.phonemes[0].tajweed == ["qalqalah"]
    assert result.phonemes[1].tajweed == ["madd"]
    assert result.phonemes[2].tajweed == ["shadda", "ghunnah"]
    assert result.qiraah == "warsh"


@pytest.mark.parametrize("qiraah", ["hafs", "warsh", "qalun"])
def test_reading_specific_phonemes_are_preserved(qiraah):
    # The caller supplies the reviewed reading-specific reference, not a
    # heuristic letter substitution derived from the reading's name.
    result = align(reference(("ʕ", "aː"), qiraah), acoustics([1, 2], labels=("_", "ʕ", "aː")))
    assert [p.symbol for p in result.phonemes] == ["ʕ", "aː"]
    assert result.qiraah == qiraah


@pytest.mark.parametrize("symbols", [("b",), ("b", "a"), ("b", "b"), ("b", "a", "b")])
def test_viterbi_matches_exhaustive_ctc_paths(symbols):
    values = np.random.default_rng(42).dirichlet([1, 1, 1], size=5)
    data = AcousticFrames(
        labels=["_", "b", "a"], blank_id=0, frame_shift_ms=20, model_id="random fixture", probabilities=values.tolist()
    )
    tokens = [{"b": 1, "a": 2}[s] for s in symbols]
    best = -math.inf
    for path in itertools.product(range(3), repeat=5):
        collapsed = [token for i, token in enumerate(path) if token != 0 and (i == 0 or token != path[i - 1])]
        if collapsed == tokens:
            best = max(best, sum(math.log(values[i, token]) for i, token in enumerate(path)))
    result = align(reference(symbols), data)
    assert result.path_log_probability == pytest.approx(best)
    assert all(0 < phone.confidence <= 1 for phone in result.phonemes)


def test_confidence_tracks_evidence_and_is_not_fabricated():
    config = dict(labels=["_", "b", "a"], blank_id=0, frame_shift_ms=20, model_id="fixture")
    high = align(reference(), AcousticFrames(**config, probabilities=[[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]))
    low = align(reference(), AcousticFrames(**config, probabilities=[[0.45, 0.1, 0.45], [0.45, 0.45, 0.1]]))
    assert high.confidence == pytest.approx(0.9)
    assert low.confidence == pytest.approx(0.1)
    assert "not calibrated" in high.warnings[0]


@pytest.mark.parametrize("split", range(1, 8))
def test_streaming_matches_batch_exactly_for_any_chunk_boundary(split):
    data = acoustics([0, 1, 1, 0, 2, 2, 2, 0])
    decoder = CTCAligner(reference(), data)
    decoder.append(FrameBatch(probabilities=data.probabilities[:split]))
    partial = decoder.result(final=False)
    assert not partial.final
    assert partial.frames_processed == split
    assert any("provisional" in message for message in partial.warnings)
    decoder.append(FrameBatch(probabilities=data.probabilities[split:]))
    assert decoder.result(final=True) == align(reference(), data)
    with pytest.raises(AlignmentError, match="already finalized"):
        decoder.append(FrameBatch(probabilities=data.probabilities))


def test_no_frames_or_incomplete_audio_do_not_produce_fake_alignments():
    decoder = CTCAligner(reference(), acoustics([1]))
    with pytest.raises(AlignmentError, match="No acoustic frames"):
        decoder.result(final=True)
    decoder.append(FrameBatch(probabilities=[[0, 1, 0]]))
    with pytest.raises(AlignmentError, match="No valid CTC path"):
        decoder.result(final=True)
    assert len(decoder.result(final=False).phonemes) == 1


def test_blank_unknown_phone_and_wrong_width_rejected():
    with pytest.raises(AlignmentError, match="absent"):
        CTCAligner(reference(("ʕ",)), acoustics([1]))
    with pytest.raises(AlignmentError, match="blank"):
        CTCAligner(reference(("_",)), acoustics([1]))
    decoder = CTCAligner(reference(), acoustics([1]))
    with pytest.raises(AlignmentError, match="width"):
        decoder.append(FrameBatch(probabilities=[[1, 0]]))
    assert decoder.frames_processed == 0


@pytest.mark.parametrize("matrix", [[], [[0.1, 0.1]], [[1, 0], [1, 0, 0]], [[-1, 2]], [[math.nan, 0]], [[math.inf, 0]]])
def test_invalid_frame_matrices(matrix):
    with pytest.raises(ValidationError):
        FrameBatch(probabilities=matrix)


@pytest.mark.parametrize(
    "change",
    [
        {"blank_id": 3},
        {"blank_id": True},
        {"frame_shift_ms": 0},
        {"frame_shift_ms": math.nan},
        {"offset_ms": -1},
        {"labels": ["_", "_"]},
    ],
)
def test_invalid_acoustic_configuration(change):
    data = acoustics([1]).model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        AcousticFrames.model_validate(data)


def test_reference_and_trellis_budgets(monkeypatch):
    import phoneme_alignment

    target = reference().model_dump()
    target["verses"].append(target["verses"][0])
    with pytest.raises(ValidationError, match="Duplicate"):
        AlignmentReference.model_validate(target)
    with pytest.raises(ValidationError):
        reference((), "hafs")
    with pytest.raises(ValidationError):
        reference(qiraah="invented")
    monkeypatch.setattr(phoneme_alignment, "MAX_CELLS", 5)
    decoder = CTCAligner(reference(), acoustics([1]))
    with pytest.raises(AlignmentError, match="budget"):
        decoder.append(FrameBatch(probabilities=[[0, 1, 0], [0, 0, 1]]))
    assert decoder.frames_processed == 0


def test_http_api_and_validation():
    with client() as api:
        data = {"reference": reference().model_dump(), "acoustics": acoustics([1, 2]).model_dump()}
        response = api.post("/quran/alignment/frames", json=data)
        assert response.status_code == 200
        assert len(response.json()["phonemes"]) == 2
        data["acoustics"] = acoustics([1]).model_dump()
        assert api.post("/quran/alignment/frames", json=data).status_code == 422


def test_websocket_incremental_alignment_and_disconnect():
    frames = acoustics([1, 0, 2])
    config = AcousticConfig.model_validate(frames.model_dump(exclude={"probabilities"}))
    with client() as api, api.websocket_connect("/stream") as socket:
        socket.send_json({"reference": reference().model_dump(), "acoustics": config.model_dump()})
        assert socket.receive_json() == {"status": "ready"}
        socket.send_json({"probabilities": frames.probabilities[:1]})
        assert socket.receive_json()["frames_processed"] == 1
        socket.send_json({"probabilities": frames.probabilities[1:], "final": True})
        assert socket.receive_json() == align(reference(), frames).model_dump(mode="json")
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1000


def test_websocket_invalid_input_and_connection_isolation():
    frames = acoustics([1, 2])
    config = AcousticConfig.model_validate(frames.model_dump(exclude={"probabilities"}))
    with client() as api:
        with api.websocket_connect("/stream") as socket:
            socket.send_json({"reference": reference().model_dump(), "acoustics": config.model_dump()})
            socket.receive_json()
            socket.send_json({"probabilities": [[0, 0, 0]]})
            assert "error" in socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as error:
                socket.receive_json()
            assert error.value.code == 1008
        with api.websocket_connect("/stream") as socket:
            socket.send_json({"reference": reference().model_dump(), "acoustics": config.model_dump()})
            socket.receive_json()
            socket.send_json({"probabilities": frames.probabilities, "final": True})
            assert socket.receive_json()["frames_processed"] == 2


def test_main_routes_require_service_credentials(monkeypatch):
    import main

    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SERVICE_API_KEY", "backend-key")
    with TestClient(main.app) as api:
        data = {"reference": reference().model_dump(), "acoustics": acoustics([1, 2]).model_dump()}
        assert api.post("/quran/alignment/frames", json=data).status_code == 401
        assert api.post("/quran/alignment/frames", json=data, headers={"X-API-Key": "backend-key"}).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with api.websocket_connect("/quran/alignment/stream"):
                pass
        with api.websocket_connect("/quran/alignment/stream", headers={"X-API-Key": "backend-key"}) as socket:
            frames = acoustics([1, 2])
            socket.send_json(
                {
                    "reference": reference().model_dump(),
                    "acoustics": AcousticConfig.model_validate(
                        frames.model_dump(exclude={"probabilities"})
                    ).model_dump(),
                }
            )
            assert socket.receive_json() == {"status": "ready"}


def test_corpus_processing_keeps_line_order_and_records_individual_failures():
    import io
    import json

    from scripts.align_quran import align_corpus

    request = json.dumps({"reference": reference().model_dump(), "acoustics": acoustics([1, 2]).model_dump()})
    output = io.StringIO()
    assert align_corpus([request, "not JSON", request], output) == 1
    results = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["line"] for record in results] == [1, 2, 3]
    assert "error" in results[1]
    assert results[0]["result"] == results[2]["result"]
