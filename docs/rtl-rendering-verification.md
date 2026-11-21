# RTL and Mixed-Script Rendering Verification

This document is the release evidence for issue #878. It defines how
right-to-left and mixed-script answers are verified at the browser level and
records the sacred-text reviewer sign-off.

## Scope and limitation

This repository is the **API service only**. There is no production web client,
no `static/` directory, no templates, and no JavaScript application in this
repository, so the production client cannot be driven from here.

The honest equivalent is the committed reference client in
[`tests/browser/`](../tests/browser/):

- `index.html`, `styles.css`, and `app.js` have no build step and load no CDN
  assets, external fonts, or network resources.
- It consumes fixture JSON shaped like the real `ChatResponse`: `response`,
  `language`, `requested_language`, `direction`, and `citations`, each with a
  `reference` and an Arabic-script `quote`.
- It sets `dir` from `direction`, renders citation links next to their quote,
  and exposes the copied-text payload.

Passing this suite proves the documented contract renders correctly in
Chromium. It does not prove a specific production client implements the
contract. The reviewer checklist below must be completed against the real
client before a release that changes multilingual rendering.

## Rendering test matrix

The suite runs each case at desktop (1280x800) and mobile (390x844).

| Case | Script | Direction | Citation checks | Assertions |
|---|---|---|---|---|
| Arabic | Arabic with an embedded Latin word | rtl | 2:153, 103:3 | `dir` and computed direction, bidi run order, `؟`/`،` intact, citation adjacency, copied text, aria/tab order, screenshot baseline |
| Urdu | Arabic (Nastaliq) with an embedded Latin word | rtl | 2:153 | same as Arabic |
| Swahili-English | Latin mixed-language | ltr | 2:153 with Arabic quote | `dir` and computed direction, Latin run order, `.` intact, citation adjacency, copied text, aria/tab order, screenshot baseline |
| Mixed | Arabic + English in one answer | rtl | 2:153, 103:3 | same as Arabic |

The bidi assertions do not trust CSS. They read `Range.getBoundingClientRect()`
for the first and last logical runs and for an embedded Latin word, and assert
the visual x-order the Unicode bidi algorithm should produce. Arabic
punctuation is measured the same way.

## Running the suite

```bash
pip install -r requirements-dev.txt
python -m playwright install --with-deps chromium
pytest -q tests/browser
```

If Playwright or Chromium is missing, the suite **skips** with an explicit
message. It is also excluded from the default `pytest tests/` session
(`norecursedirs` in `pytest.ini`), because the sync Playwright API leaves an
event loop running that breaks later sync `TestClient` tests. CI installs
Chromium and runs `pytest tests/browser` in its own `browser-rtl` job.

Outputs:

- `tests/browser/artifacts/<case>_<viewport>.png`: current screenshots (CI
  artifact, gitignored).
- `tests/browser/artifacts/traces/<case>_<viewport>.zip`: Playwright trace saved
  only when a case fails.
- `tests/browser/baselines/`: committed visual baselines.

## Visual regression policy

Screenshots are compared with a Pillow pixel diff. More than 5% of pixels
changing fails the test. This tolerance absorbs operating-system font
rasterisation differences while still catching a direction, layout, or citation
placement regression.

To update baselines:

```bash
pytest -q tests/browser --update-baselines
```

This rewrites the files in `tests/browser/baselines/`. It is never run
automatically. Updating a baseline is a visual-contract change and requires an
explicit human reviewer to inspect the before/after images and sign the table
below.

## Sacred-text reviewer sign-off

Complete this table against the production client before releasing a
multilingual or sacred-text rendering change. Attach the screenshots or traces
used for each row.

| Check | Desktop | Mobile | Reviewer | Date |
|---|---|---|---|---|
| Arabic answer sets RTL direction | [ ] | [ ] | | |
| Urdu answer sets RTL direction | [ ] | [ ] | | |
| Mixed Arabic/English keeps Arabic runs RTL and Latin runs LTR | [ ] | [ ] | | |
| Swahili-English stays LTR and remains readable | [ ] | [ ] | | |
| `؟` and `،` are not reordered or mirrored | [ ] | [ ] | | |
| Every citation link maps to the right `surah:ayah` and sits next to its quote | [ ] | [ ] | | |
| Quran quotations remain in Arabic script with a translation | [ ] | [ ] | | |
| Copied text equals the displayed text | [ ] | [ ] | | |
| Screen-reader reading order is logical | [ ] | [ ] | | |
| Keyboard focus order follows reading order in RTL layout | [ ] | [ ] | | |
| No clipping, overlap, or horizontal scroll at the tested viewport | [ ] | [ ] | | |

Reviewer name: ____________________  Signature: ____________________

## API contract this suite depends on

- `ChatResponse.language`: effective BCP-47 base code or `null`.
- `ChatResponse.requested_language`: normalized requested code, preserved
  across provider retries and experiment fallback.
- `ChatResponse.direction`: `"rtl"` or `"ltr"`, from
  [`language_direction.py`](../language_direction.py).
- Streaming `metadata` and `done` events carry the same three fields.
- `citations[]` entries carry a validated `reference` and the quoted text.

The language support matrix lives in
[`docs/language-support-matrix.md`](language-support-matrix.md).
