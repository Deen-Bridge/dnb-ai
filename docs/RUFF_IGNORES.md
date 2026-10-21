# Ruff per-file ignores

Every per-file ignore must name the rule and explain why it is necessary. When
the underlying violation is fixed, remove the suppression in the same change;
unrelated generated, benchmark, and migration files should remain explicitly
scoped rather than adding broad directory ignores.
