# Retrieval logging

Retrieval and citation modules use the project logger with structured context
instead of direct `print()` calls. Logs may include operation identifiers and
counts, but must exclude prompts, credentials, document contents, and provider
secrets.
