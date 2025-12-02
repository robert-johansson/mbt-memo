"""
Mentalizing Example in memo

Demonstrates theory-of-mind reasoning and how different prior beliefs
affect social inference - relevant to understanding personality disorders.

Scenario: Jane observes Alex's behavior (delayed text reply) and
infers Alex's intention. Different attachment styles lead to different priors.
"""

import jax
import jax.numpy as np
from memo import memo

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Intentions: 0=busy, 1=caring, 2=rejecting
Intentions = [0, 1, 2]
intention_labels = ['busy', 'caring', 'rejecting']

# Behaviors: 0=quick, 1=moderate, 2=long delay
Behaviors = [0, 1, 2]
behavior_labels = ['quick', 'moderate', 'long delay']


@jax.jit
def likelihood(behavior, intention):
    matrix = np.array([
        [0.3, 0.5, 0.2],   # busy: mostly moderate
        [0.7, 0.25, 0.05], # caring: mostly quick
        [0.1, 0.3, 0.6],   # rejecting: mostly long
    ])
    return matrix[intention, behavior]


@jax.jit
def secure_prior_weight(intention):
    weights = np.array([0.5, 0.3, 0.2])
    return weights[intention]


@jax.jit
def anxious_prior_weight(intention):
    weights = np.array([0.1, 0.1, 0.8])
    return weights[intention]


# Simple prior distributions
@memo
def secure_prior[i: Intentions]():
    alex: given(intention in Intentions, wpp=secure_prior_weight(intention))
    return E[alex.intention == i]


@memo
def anxious_prior[i: Intentions]():
    alex: given(intention in Intentions, wpp=anxious_prior_weight(intention))
    return E[alex.intention == i]


# Inference with secure attachment
@memo
def secure_inference[i: Intentions, b: Behaviors]():
    cast: [alex, jane]
    jane: thinks[
        alex: given(intention in Intentions, wpp=secure_prior_weight(intention)),
        alex: given(behavior in Behaviors, wpp=likelihood(behavior, intention))
    ]
    jane: knows(i)
    jane: observes [alex.behavior] is b
    return jane[E[alex.intention == i]]


# Inference with anxious attachment
@memo
def anxious_inference[i: Intentions, b: Behaviors]():
    cast: [alex, jane]
    jane: thinks[
        alex: given(intention in Intentions, wpp=anxious_prior_weight(intention)),
        alex: given(behavior in Behaviors, wpp=likelihood(behavior, intention))
    ]
    jane: knows(i)
    jane: observes [alex.behavior] is b
    return jane[E[alex.intention == i]]


if __name__ == "__main__":
    print("=" * 60)
    print("Mentalizing Example: Inferring intentions from behavior")
    print("=" * 60)

    print("\n--- Prior beliefs (before observation) ---")
    print(f"Secure attachment prior:  {secure_prior()}")
    print(f"Anxious attachment prior: {anxious_prior()}")

    print("\n--- Full inference matrices ---")
    print("Secure inference [intention x behavior]:")
    print(secure_inference())
    print("\nAnxious inference [intention x behavior]:")
    print(anxious_inference())

    observed = 2  # long delay
    print(f"\n--- After observing: {behavior_labels[observed]} ---")
    secure_post = secure_inference()[:, observed]
    anxious_post = anxious_inference()[:, observed]
    print(f"Secure posterior:  {secure_post}")
    print(f"Anxious posterior: {anxious_post}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(intention_labels))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, secure_prior(), width, label='Secure', color='steelblue')
    ax.bar(x + width/2, anxious_prior(), width, label='Anxious', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(intention_labels)
    ax.set_ylabel('Probability')
    ax.set_title('Prior beliefs')
    ax.legend()

    ax = axes[1]
    ax.bar(x - width/2, secure_post, width, label='Secure', color='steelblue')
    ax.bar(x + width/2, anxious_post, width, label='Anxious', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(intention_labels)
    ax.set_ylabel('Probability')
    ax.set_title(f'Posterior after "{behavior_labels[observed]}"')
    ax.legend()

    ax = axes[2]
    im = ax.imshow(anxious_inference(), cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(behavior_labels)))
    ax.set_xticklabels(behavior_labels)
    ax.set_yticks(range(len(intention_labels)))
    ax.set_yticklabels(intention_labels)
    ax.set_xlabel('Observed behavior')
    ax.set_ylabel('Inferred intention')
    ax.set_title('Anxious inference matrix')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('examples/mentalizing_results.png', dpi=150)
    print("\nSaved visualization to examples/mentalizing_results.png")
