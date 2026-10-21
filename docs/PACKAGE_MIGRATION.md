# Python package migration

Moving root modules into packages must preserve import compatibility during the
migration. Each moved module needs an explicit package owner, updated imports,
and focused tests; circular dependencies should be resolved at package
boundaries rather than hidden with runtime imports.
