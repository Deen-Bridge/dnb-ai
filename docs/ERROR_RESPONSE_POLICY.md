# Safe API error responses

API error responses expose a stable public error code and request correlation
identifier, while stack traces, provider errors, prompts, and exception details
remain server-side. Unexpected failures map to a generic 500 response and are
logged with the correlation identifier for operators.
