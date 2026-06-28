# Contributing to Deen Bridge AI Service

Thank you for your interest in contributing to Deen Bridge! This is the AI service that powers intelligent features across the platform.

## Drips Wave Program

This repository participates in the **Stellar Drips Wave** bounty program. Contributors can earn rewards by completing issues tagged with Wave labels.

### How It Works

1. **Find an Issue**: Look for issues with `wave:X` labels (where X is the point value)
2. **Claim the Issue**: Comment on the issue to express interest
3. **Submit a PR**: Complete the work and submit a pull request
4. **Earn Points**: Once merged, you earn points that translate to rewards

### Point Labels

| Label | Points | Typical Scope |
|-------|--------|---------------|
| `wave:1` | 1 point | Documentation, typos, small fixes |
| `wave:2` | 2 points | Bug fixes, minor improvements |
| `wave:3` | 3 points | New features, optimizations |
| `wave:4` | 4 points | Complex features, major changes |

## Getting Started

### Prerequisites

1. Python 3.10 or higher
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

# Create .env file
echo "GOOGLE_AI_API_KEY=your_api_key_here" > .env

# Run the server
uvicorn main:app --reload
```

### Making Changes

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Test your changes locally

4. Commit with a descriptive message:
   ```bash
   git commit -m "feat: improve response caching"
   ```

5. Push and create a PR:
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

1. **Title**: Use conventional commit format
2. **Description**: Explain what and why
3. **Link Issue**: Reference the issue number (`Closes #123`)
4. **Testing**: Describe how you tested the changes

## Code of Conduct

1. Be respectful and inclusive
2. Welcome newcomers
3. Focus on constructive feedback
4. Follow Islamic principles of brotherhood

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
