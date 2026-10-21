# Nested summarization

Long contexts should be summarized in bounded chunks, then combined through a
second pass that preserves source order and provenance. The reducer must handle
empty chunks and provider failures without silently dropping the remaining
context.
