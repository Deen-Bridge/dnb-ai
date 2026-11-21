# Language Support Matrix

This is the release evidence for issues #879 (advertised language modules) and
#878 (direction metadata and rendered-text verification). It is the single
public table for what the AI service advertises, where each language is
implemented, and what is deliberately **not** implemented.

## Scope and honest limitations

- This repository is the API service only. There is no production web client,
  so client behaviour is verified against the committed reference harness in
  [`tests/browser/`](../tests/browser/) plus the documented `/chat` contract.
- Urdu and Malay/Indonesian are wired into the language whitelist, prompt
  policy, and their own HTTP routers. They are **not** wired into the `/chat`
  language heuristic: `/chat` still relies on the caller's code plus LLM
  auto-detection for those two languages.
- The Malay router exposes the existing offline engines. It does not change
  how `/chat` generates Malay answers.
- Jawi is recognised as right-to-left Arabic script; Latin-script Malay and
  Indonesian remain left-to-right.

## Matrix

| Language | Code | Script | Direction | Router | Whitelist | Chat integration | Direction/citation tests |
|---|---|---|---|---|---|---|---|
| Arabic | `ar` | Arabic | rtl | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_direction.py`, `tests/test_language_contract.py` |
| English | `en` | Latin | ltr | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_direction.py` |
| Urdu | `ur` | Arabic (Nastaliq) | rtl | `/urdu` | yes | whitelist + prompt policy | `tests/test_language_direction.py`, `tests/test_urdu_processor.py`, `tests/test_language_contract.py` |
| Malay | `ms` | Latin / Jawi | ltr / rtl | `/malay` | yes | whitelist + prompt policy | `tests/test_language_direction.py`, `tests/test_malay.py`, `tests/test_language_contract.py` |
| French | `fr` | Latin | ltr | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_direction.py` |
| Turkish | `tr` | Latin | ltr | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_contract.py` |
| Indonesian | `id` | Latin / Jawi | ltr / rtl | `/malay` | yes | whitelist + prompt policy | `tests/test_language_direction.py`, `tests/test_malay.py` |
| Bengali | `bn` | Bengali | ltr | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_contract.py` |
| Persian | `fa` | Arabic | rtl | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_direction.py`, `tests/test_language_contract.py` |
| Hausa | `ha` | Latin | ltr | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_contract.py` |
| Swahili | `sw` | Latin | ltr | `/swahili` | yes | detection + prompt enhancement + headers | `tests/test_language_direction.py`, `tests/test_swahili.py` |
| Tagalog | `tl` | Latin | ltr | none (chat only) | yes | whitelist + prompt policy | `tests/test_language_contract.py` |

The `SUPPORTED_LANGUAGES` mapping in `main.py` is the source of truth. The
README table, this matrix, and the module table in
[`docs/languages.md`](languages.md) are checked against it by
`tests/test_language_contract.py`, so adding or removing a language without
updating all three fails the suite.

## Response contract

Every non-streaming `ChatResponse` carries:

- `language`: the effective BCP-47 base code, or `null` for auto-detection.
- `requested_language`: the normalized code the caller asked for, preserved
  across provider retries and experiment fallback.
- `direction`: `"rtl"` or `"ltr"`, computed by
  [`language_direction.py`](../language_direction.py) from the effective
  language, or from the answer's dominant script when the language is unknown.

Streaming responses expose the same three fields on the `metadata` event and
recompute `direction` from the final text on the `done` event.

## Unsupported language requests

Unknown codes are **not** a validation error:

1. The code is lowercased and whitespace-trimmed.
2. The exact code is matched against `SUPPORTED_LANGUAGES`.
3. The base code is tried (`ur-PK` becomes `ur`).
4. If neither matches, a warning is logged and the request proceeds with
   auto-detection. The response then has `language: null`,
   `requested_language: null`, and `direction` derived from the answer text
   (default `"ltr"` when the text has no dominant script).

## Verification

| Check | Command |
|---|---|
| Contract and matrix agreement | `pytest -q tests/test_language_contract.py` |
| Direction helper | `pytest -q tests/test_language_direction.py` |
| Urdu engine and router | `pytest -q tests/test_urdu_processor.py` |
| Malay engine and router | `pytest -q tests/test_malay.py tests/test_malay_extended.py` |
| Browser rendering (desktop + mobile) | `pytest -q tests/browser` |

Browser rendering evidence, the desktop/mobile test matrix, and the
sacred-text reviewer sign-off live in
[`docs/rtl-rendering-verification.md`](rtl-rendering-verification.md).
