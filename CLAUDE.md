# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains work on a computational psychiatry framework that models personality disorders (PDs) as aberrant Bayesian inference using **memo**, a probabilistic programming language for theory-of-mind reasoning.

**Core Concept**: PDs are viewed as distortions in both the parameters (priors, likelihoods, precision) and the syntax/expressivity of social reasoning. The work integrates:
- ICD-11's dimensional model of personality disorders
- Mentalization-Based Treatment (MBT) framework (prementalizing modes: psychic equivalence, pretend mode, teleological stance)
- Bayesian inference models with the memo language as a "language of thought" for social cognition

## memo Language Constructs

When working with memo code or pseudocode in this project, understand these key constructs:

- `agent: chooses(...)` - represents stochastic choices tied to a specific agent
- `agent: observes [...] is ...` - models what information an agent actually receives
- `agent: thinks [...]` - defines one agent's model of others (nested frames of mind)
- `Pr[...]`, `E[...]`, `Var[...]` - probabilistic operators
- `imagine[...]`, `wants/EU`, `cost @ f(...)` - higher-level reasoning tools

**Key Constraints**:
- No "mind reading": agents can't directly access others' private choices unless observed
- No "mind control": agents must know the probabilities for their own choices
- False beliefs are explicitly representable via different frames and condition maps

## MBT → Computational Mapping

When interpreting or implementing clinical concepts:

- **Teleological mode**: shallow models without latent mental states (no ToM layer)
- **Psychic equivalence**: extreme prior precision + breakdown of Bayesian updating under affective load
- **Pretend mode**: inference decoupled from evidence (posterior ≈ prior)
- **Hypermentalizing**: overly complex hypothesis space with weak priors/poor regularization
- **Hypomentalizing**: truncated models with no mental-state variables or very noisy likelihoods

## Current Repository State

This repository is in early stages and currently contains only documentation/analysis files. There is no build system, test suite, or executable code yet.

When code is added, typical memo models will need:
- JAX for array programs and automatic differentiation
- Discrete, finite model representations for exact enumerative inference
- The `@memo` decorator and proper axis annotations for inference
