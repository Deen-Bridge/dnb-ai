# Contributing to Deen Bridge AI Service

Thank you for your interest in contributing to Deen Bridge! This is the AI service that powers intelligent features across the platform.

## Drips Wave Program

This repository participates in the **Stellar Drips Wave** bounty program. Contributors can earn rewards by resolving issues during Wave cycles. Everyone is welcome to contribute — no religious background or knowledge is required; our issues are regular engineering tasks.

### How It Works

1. **Find an Issue**: During an active Wave, browse this repo's issues in the [Drips Wave app](https://www.drips.network/wave)
2. **Apply**: Apply to work on the issue through the Drips Wave app; the maintainer reviews applications and assigns one contributor
3. **Submit a PR**: Complete the work and open a pull request (base branch `dev`) before the Wave ends
4. **Earn Points**: Once the issue is marked resolved during the Wave, you earn its Points, which convert to rewards from the Wave pool

### Complexity & Points

Points are assigned per issue by the maintainer in the Drips Wave dashboard using Drips' three complexity tiers:

| Complexity | Points | Typical Scope                              |
| ---------- | ------ | ------------------------------------------ |
| Trivial    | 100    | Typos, small bug fixes, minor copy changes |
| Medium     | 150    | Standard features or involved bug fixes    |
| High       | 200    | Complex features, refactors, integrations  |

Issues carry `complexity:trivial`, `complexity:medium`, or `complexity:high` labels that mirror these tiers.

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
