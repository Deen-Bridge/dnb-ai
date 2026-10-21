# Quran phoneme alignment

The alignment service finds the most likely CTC path between acoustic model
posteriors and a supplied pronunciation reference. It returns each phoneme's
start/end time, word and verse boundaries, acoustic confidence, reading,
reciter, model identifier, and the complete path log probability. It works with
phonemes of any script, including Arabic articulation and Tajweed distinctions
represented by a compatible model's vocabulary.

The reference explicitly supplies words and their phonemes for `hafs`, `warsh`,
or `qalun`. Use an expert reviewed reference in the acoustic model's alphabet,
including the selected reading's madd, shadda, nasalization, assimilation and
pause forms. A reading name alone cannot determine those pronunciations. The
service does not ship a canonical Quran pronunciation lexicon or automatically
infer all Tajweed rules from unvocalized Arabic text. It preserves reference
annotations and aligns their acoustic evidence; it does not grade religious
correctness or replace the separate recitation quality scorer.

## Frame API

Send `POST /quran/alignment/frames` with `X-API-Key` and this structure:

```json
{
  "reference": {
    "qiraah": "hafs",
    "reciter": "reference reciter",
    "verses": [{
      "surah": 1,
      "ayah": 1,
      "words": [{
        "text": "بِسْمِ",
        "phonemes": [{"symbol":"b"}, {"symbol":"i"}, {"symbol":"s"}, {"symbol":"m"}, {"symbol":"i"}]
      }]
    }]
  },
  "acoustics": {
    "labels": ["_", "b", "i", "s", "m"],
    "blank_id": 0,
    "frame_shift_ms": 20,
    "offset_ms": 0,
    "model_id": "your pinned Arabic phoneme checkpoint",
    "probabilities": [
      [0.02,0.92,0.02,0.02,0.02],
      [0.02,0.02,0.92,0.02,0.02],
      [0.02,0.02,0.02,0.92,0.02],
      [0.02,0.02,0.02,0.02,0.92],
      [0.02,0.02,0.92,0.02,0.02]
    ]
  }
}
```

The probabilities above are a **synthetic request example**, not a recording.
Real posteriors must come from an acoustic phoneme model, never from the target
text. Rows are normalized distributions over `labels`; zero probabilities are
supported. The decoder uses log probabilities and CTC's stay, advance, and skip
transitions. Adjacent identical phonemes require an intervening blank. Unknown
reference labels, impossible paths, and oversized inputs return 422 rather than
inventing timestamps. Each `Phone` can optionally carry `tajweed` tags: `madd`,
`ghunnah`, `shadda`, `ikhfa`, `idgham`, `iqlab`, `qalqalah`, `tafkhim`, `tarqiq`.

Times are derived from model frame indices, frame stride and the optional audio
offset. Silence can create gaps between phonemes. Word/verse boundaries enclose
their phonemes. Confidence per phoneme is the geometric mean of that label's
posterior over its assigned frames; word, verse and overall scores take the
geometric mean of their phoneme scores. These are **uncalibrated acoustic
scores**, not a probability that the recitation or Tajweed is correct. A forced
alignment can fit an incorrect reference with low confidence. Model resolution,
CTC peaks, and acoustic quality limit boundary precision.

## Optional audio recognition

Install the regular dependencies and `pip install -r requirements-alignment.txt`.
Configure `ALIGNMENT_MODEL_ID`, a full 40-character checkpoint commit SHA in
`ALIGNMENT_MODEL_REVISION`, and `ALIGNMENT_MODEL_QIRAAT` (a comma-separated list
of the readings that checkpoint actually supports). Model and tokenizer load
lazily, without remote repository code, and are cached per worker. The adapter
supports standard Transformers `AutoModelForCTC` checkpoints with an atomic
phoneme vocabulary, a pad/blank ID and `inputs_to_logits_ratio`. A character ASR
checkpoint is insufficient for phoneme alignment. The model's 16 kHz feature
extractor and tokenizer must be included with it.

The [Meta phoneme recognition checkpoint](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
is an example of this model interface. It is not evidence of Quran-specific
accuracy. Select and validate an appropriately licensed Arabic/Quran phoneme
checkpoint for your application and reciters before deployment.

Send multipart `POST /quran/alignment/audio` with a `reference` JSON field in the
same format as above and an `audio` file. Audio must be mono, 16 kHz, PCM16 WAV,
0.1–60 seconds, up to 2 MB. The adapter extracts real frame posteriors and feeds
them into the same decoder. Missing optional dependencies/configuration,
unsupported model readings, model load or inference failures return 503;
malformed audio/reference or incompatible phoneme targets return 422. The
service never falls back to guessed phoneme timing. Acoustic inference is
serialized per worker to bound CPU/memory use. Production must provision model
weights, memory and CPU/GPU capacity separately from the lightweight API.

## Live and corpus processing

Connect to `/quran/alignment/stream` over WebSocket with the service `X-API-Key`
header. The trusted backend should proxy browser microphone streams; do not put
the service secret in browser code.

1. Send `{"reference": {...}, "acoustics": {...}}`, where acoustics contains
   labels, blank ID, stride, offset and model ID, but no probabilities.
2. Wait for `{"status":"ready"}`.
3. Send consecutive, non-overlapping batches of normalized `probabilities`.
   Each reply is a provisional alignment with `final: false`. Earlier partial
   boundaries may change; clients should replace their prior preview.
4. Include `"final": true` with the last nonempty batch. The final response
   has complete reference coverage and the server closes normally (1000).
   Invalid input closes with 1008. Disconnection discards connection-local state.

A streaming acoustic frontend must handle its own overlapping inference windows
and supply each new frame exactly once. This endpoint incrementally decodes new
**model frames**; it does not run the optional batch Wav2Vec2 recognizer as a
streaming microphone model. Per connection, limits are 512 reference phonemes,
6,000 frames, 512 acoustic labels, and 3,000,000 trellis cells. Split longer
recordings at verse boundaries and carry the global `offset_ms`. Rolling score
vectors and byte-sized traceback steps bound decoder resource use.

For corpus processing, write one frame API request per JSONL line and run:

```bash
python -m scripts.align_quran requests.jsonl alignments.jsonl
```

Output preserves input order and includes each line's result or error. The CLI
streams records, flushes output for recovery, and exits 1 if any record failed.

## Verification and accuracy limits

`pytest -q tests/test_phoneme_alignment.py tests/test_phoneme_model.py` checks CTC
paths against exhaustive enumeration, repeated-label constraints, variable
length boundaries, stream/batch equivalence, readings/Tajweed references,
confidence, invalid inputs, authentication, corpus errors, WAV decoding, and the
pinned model adapter using fake inference objects. No real model is downloaded
by offline tests.

`python -m scripts.bench_phoneme_alignment` measures streaming decoder throughput
on 60 seconds of synthetic posteriors and reports real time factor, excluding
acoustic inference. It is a throughput check, **not an accuracy benchmark**.

No empirical Quran phoneme boundary accuracy or end-to-end live inference
latency is claimed by these tests. Validate on expert timestamped, held-out
recitations for each deployed reading, reciter, and acoustic condition; measure
phoneme onset/offset error, alignment coverage and confidence calibration.
Accurate deployment depends on both the selected model and the reviewed
pronunciation references. The CTC algorithm follows the
[forced alignment approach documented by PyTorch](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html),
implemented here with NumPy to keep the base service lightweight.
