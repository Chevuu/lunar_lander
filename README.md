# Evolving a Lunar Lander with Genetic Programming

This workspace contains the deliverable for the Evolutionary Algorithms course project. We evolved a symbolic controller for the `LunarLander-v2` environment using differentiable Genetic Programming, achieving a test score of **6617** over 30 episodes (87% survival rate).

## What to look at

Open **`best_model_showcase.ipynb`** — it walks through everything:

- The seven improvements made over the baseline (hyperparameter sweep, extended operators, elitism, seeding, fitness function penalties, validation-based model selection, coefficient gating)
- The best evolved model and its symbolic formula
- Training dynamics plots and a comparison against the baseline
- GIFs showing the agent's progression across generations and performance across varied seeds

## Setup

Requires Python < 3.10.

```bash
pip install -r requirements.txt
```
