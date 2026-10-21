# Configuration validation coverage

Configuration tests should cover required fields, invalid values, defaults,
environment overrides, and readable error messages. Fixtures must avoid real
credentials and should assert that invalid configuration is rejected before an
agent or provider is initialized.
