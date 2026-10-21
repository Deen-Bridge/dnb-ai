# Application debug output

Application and library modules must use the project logger rather than
`print()` so production output is structured, configurable, and testable.
Benchmark and command-line scripts may retain direct output when it is their
documented interface.
