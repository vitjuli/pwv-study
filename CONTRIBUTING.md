# Contributing to pwv-study

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/vitjuli/pwv-study.git
cd pwv-study
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
pip install pytest
```

3. Run tests to verify your setup:
```bash
pytest -q
```

## Running the Project

### Generate synthetic data:
```bash
pwv-study gen --seconds 30 --fs 1000 --hr 70 --seed 42 --out results/demo
```

### Run pipelines:
```bash
pwv-study run --in_csv results/demo.csv --out_dir results/study_demo
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting changes
- Run `pytest -q` to execute the test suite

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure nothing breaks
5. Commit your changes with clear, descriptive messages
6. Push to your fork
7. Submit a pull request

## Questions?

Feel free to open an issue for questions or discussions about the project.
