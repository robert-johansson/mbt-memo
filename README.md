# mbt-memo

Computational models of social cognition in personality disorders using [memo](https://github.com/kach/memo), a probabilistic programming language for theory-of-mind reasoning.

## Overview

This project implements Mentalization-Based Treatment (MBT) concepts as Bayesian inference models. Personality disorders are modeled as distortions in:
- **Priors**: Biased baseline beliefs about others' mental states
- **Likelihood weighting**: How evidence updates beliefs
- **Model structure**: Depth and accuracy of mental state reasoning

## Examples

| File | Scenario | Key Finding |
|------|----------|-------------|
| `bpd_abandonment.py` | BPD fear of abandonment when partner doesn't reply | BPD: 81% rejection belief vs Secure: 29% |
| `npd_criticism.py` | NPD interpretation of mild criticism | NPD: 75% envy attribution vs Realistic: 25% |
| `mbt_prementalizing_modes.py` | Comparison of healthy vs pathological mentalizing | Demonstrates psychic equivalence, pretend mode, hypermentalizing |

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install memo-lang matplotlib
```

## Usage

```bash
source venv/bin/activate
python examples/bpd_abandonment.py
```

Each example produces console output and saves a visualization to `examples/*.png`.

## Dependencies

- [memo-lang](https://pypi.org/project/memo-lang/) - Probabilistic programming for ToM
- JAX - Automatic differentiation and JIT compilation
- matplotlib - Visualization

## References

- [memo Handbook](https://github.com/kach/memo)
- Bateman & Fonagy - Mentalization-Based Treatment
