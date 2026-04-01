# claude-abtest

> End-to-end A/B test analysis pipeline, built with assistance from Claude Code.

## Overview

This repository provides a full A/B testing analysis workflow in Python — from raw experiment data through statistical testing, reporting, and interpretation. It's designed as a reference implementation showing how Claude Code can assist with data-driven decision making workflows.

## Repository Structure

```
claude-abtest/
├── notebooks/      # Jupyter notebooks for exploratory analysis and walkthroughs
├── reports/        # Generated analysis outputs and visualizations
├── src/            # Core analysis modules (data loading, stats, metrics)
├── tests/          # Unit tests
├── CLAUDE.md       # Claude Code project context and conventions
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/cqcq-git/claude-abtest.git
cd claude-abtest
pip install -r requirements.txt
```

### Running the Analysis

Open the notebooks for a guided walkthrough:

```bash
jupyter notebook notebooks/
```

Or run the analysis directly via the `src/` modules:

```bash
python src/analyze.py
```

### Running Tests

```bash
pytest tests/
```

## What's Covered

A typical A/B test analysis pipeline includes:

- **Data ingestion** — loading experiment assignment and event data
- **Sanity checks** — verifying sample ratio mismatch (SRM) and pre-experiment balance
- **Metric computation** — calculating conversion rates, means, and other KPIs per variant
- **Statistical testing** — t-tests, z-tests, or non-parametric alternatives with p-values and confidence intervals
- **Reporting** — structured outputs summarizing results and recommendations

## Usage with Claude Code

This project includes a `CLAUDE.md` that gives Claude Code context about the analysis conventions and codebase structure. To work on this repo with Claude Code:

```bash
cd claude-abtest
claude
```

Claude will read `CLAUDE.md` automatically and be ready to help extend the pipeline, debug issues, or add new statistical tests.

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you'd like to change. Make sure to update tests as appropriate.

## License

This project is intended as example/reference code. See the repository for license details.
