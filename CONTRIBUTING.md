# Contributing to Deen Bridge AI Service

Thank you for your interest in contributing to Deen Bridge! This is the AI service that powers intelligent features across the platform.

## Contribution Workflow

1. Find or open an issue describing the change.
2. Create a focused branch from `dev` and implement the change.
3. Run the relevant checks described below.
4. Open a pull request against `dev`, link the issue, and summarize the changes and checks.

Everyone is welcome to contribute. No religious background or prior Islamic studies knowledge is required for regular engineering tasks.

## Getting Started

### Prerequisites

1. Python 3.11 or higher
2. pip package manager
3. Google AI API key (for Gemini)

### Setup

```bash
# Fork and clone the repository
git clone git@github.com:YOUR_USERNAME/dnb-ai.git
cd dnb-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the lint/type/test tooling CI runs
pip install -r requirements-dev.txt

# Create .env file from template
cp .env.example .env

# Edit .env and set your GEMINI_API_KEY

# Run the server
uvicorn main:app --reload
```

## Branching Strategy

| Branch | Purpose                                                      |
| ------ | ------------------------------------------------------------ |
| `main` | Stable, production-ready code — releases only                |
| `dev`  | Active development — **all pull requests must target `dev`** |

Maintainers periodically merge `dev` into `main` for releases. Pull requests opened against `main` will be asked to retarget `dev`.

### Making Changes

1. Create a branch from the latest `dev`:

   ```bash
   git fetch origin
   git checkout -b feature/your-feature-name origin/dev
   ```

2. Make your changes following our coding standards

3. Test your changes locally

4. Commit with a descriptive message:

   ```bash
   git commit -m "feat: improve response caching"
   ```

5. Push and create a PR **with `dev` as the base branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style

1. Follow PEP 8 guidelines
2. Use type hints for function parameters and return values
3. Write docstrings for functions and classes
4. Keep functions focused and small

Linting, formatting, and type checking are enforced in CI by [ruff](https://docs.astral.sh/ruff/)
and [mypy](https://mypy.readthedocs.io/), configured in `pyproject.toml` (line length 120).
Run these locally before pushing — CI runs exactly the same commands:

```bash
ruff check .          # lint (pycodestyle, pyflakes, import sorting, pyupgrade, bugbear)
ruff format .         # format the code (use `ruff format --check .` to only verify)
mypy .                # type check
```

`ruff check --fix .` applies the fixes ruff can make automatically.

`main.py` is type-checked with `disallow_untyped_defs`: every function there needs
parameter and return annotations. The rest of the tree is checked less strictly for now —
please annotate new code anyway.

Optionally, install the git hook so the same checks run before each commit:

```bash
pip install pre-commit
pre-commit install
```

### API Design

1. Use proper HTTP status codes
2. Return consistent JSON response formats
3. Handle errors gracefully with informative messages

### Commits

We follow Conventional Commits:

1. `feat:` for new features
2. `fix:` for bug fixes
3. `docs:` for documentation changes
4. `refactor:` for code refactoring
5. `perf:` for performance improvements
6. `test:` for adding tests

## Pull Request Guidelines

1. **Base Branch**: open the PR against `dev`, never `main`
2. **Title**: Use conventional commit format
3. **Description**: Explain what and why
4. **Link Issue**: Reference the issue number (`Closes #123`)
5. **Testing**: Describe how you tested the changes

## Code of Conduct

1. Be respectful and inclusive
2. Welcome newcomers
3. Focus on constructive feedback
4. Contributors of all backgrounds and faiths are welcome

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
