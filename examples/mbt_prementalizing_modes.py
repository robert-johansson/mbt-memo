"""
MBT Prementalizing Modes in memo

Implements the three prementalizing modes from Mentalization-Based Treatment:
1. Teleological mode: Focus on outcomes only, no mental state inference
2. Psychic equivalence: Internal feeling = external reality
3. Pretend mode: Rich mental activity decoupled from reality

Based on Table 1 from frontiers.tex.
"""

import jax
import jax.numpy as np
from memo import memo

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Friend's mental states: 0=doesn't care, 1=neutral, 2=cares
MentalState = [0, 1, 2]
mental_labels = ['doesnt_care', 'neutral', 'cares']

# Observable actions: 0=forgot birthday, 1=sent text, 2=threw party
Action = [0, 1, 2]
action_labels = ['forgot_birthday', 'sent_text', 'threw_party']


@jax.jit
def balanced_prior(mental_state):
    weights = np.array([0.2, 0.4, 0.4])
    return weights[mental_state]


@jax.jit
def action_likelihood(action, mental_state):
    # rows = mental_state, cols = action
    matrix = np.array([
        [0.6, 0.3, 0.1],   # doesn't care
        [0.3, 0.5, 0.2],   # neutral
        [0.1, 0.4, 0.5],   # cares
    ])
    return matrix[mental_state, action]


@jax.jit
def psychic_equiv_prior_abandoned(mental_state):
    weights = np.array([0.95, 0.04, 0.01])
    return weights[mental_state]


@jax.jit
def psychic_equiv_prior_loved(mental_state):
    weights = np.array([0.01, 0.04, 0.95])
    return weights[mental_state]


@jax.jit
def hypermentalizing_prior(mental_state):
    weights = np.array([0.45, 0.10, 0.45])
    return weights[mental_state]


# =============================================================================
# HEALTHY MENTALIZING
# =============================================================================

@memo
def healthy_mentalizing[m: MentalState, a: Action]():
    # Balanced prior + proper Bayesian updating
    cast: [friend, self]
    self: thinks[
        friend: given(mental_state in MentalState, wpp=balanced_prior(mental_state)),
        friend: given(action in Action, wpp=action_likelihood(action, mental_state))
    ]
    self: observes [friend.action] is a
    self: chooses(guess in MentalState, wpp=E[friend.mental_state == guess])
    return E[self.guess == m]


# =============================================================================
# PSYCHIC EQUIVALENCE - Feeling Abandoned
# =============================================================================

@memo
def psychic_equiv_abandoned[m: MentalState, a: Action]():
    # Prior overwhelmed by feeling of abandonment
    cast: [friend, self]
    self: thinks[
        friend: given(mental_state in MentalState, wpp=psychic_equiv_prior_abandoned(mental_state)),
        friend: given(action in Action, wpp=action_likelihood(action, mental_state))
    ]
    self: observes [friend.action] is a
    self: chooses(guess in MentalState, wpp=E[friend.mental_state == guess])
    return E[self.guess == m]


# =============================================================================
# PSYCHIC EQUIVALENCE - Feeling Loved
# =============================================================================

@memo
def psychic_equiv_loved[m: MentalState, a: Action]():
    # Prior overwhelmed by feeling of being loved
    cast: [friend, self]
    self: thinks[
        friend: given(mental_state in MentalState, wpp=psychic_equiv_prior_loved(mental_state)),
        friend: given(action in Action, wpp=action_likelihood(action, mental_state))
    ]
    self: observes [friend.action] is a
    self: chooses(guess in MentalState, wpp=E[friend.mental_state == guess])
    return E[self.guess == m]


# =============================================================================
# PRETEND MODE - No Reality Testing
# =============================================================================

@memo
def pretend_mode[m: MentalState]():
    # Rich belief generation but no observation/evidence use
    friend: given(mental_state in MentalState, wpp=balanced_prior(mental_state))
    return E[friend.mental_state == m]


# =============================================================================
# HYPERMENTALIZING
# =============================================================================

@memo
def hypermentalizing[m: MentalState, a: Action]():
    # Over-interpretation - extreme hypotheses favored
    cast: [friend, self]
    self: thinks[
        friend: given(mental_state in MentalState, wpp=hypermentalizing_prior(mental_state)),
        friend: given(action in Action, wpp=action_likelihood(action, mental_state))
    ]
    self: observes [friend.action] is a
    self: chooses(guess in MentalState, wpp=E[friend.mental_state == guess])
    return E[self.guess == m]


if __name__ == "__main__":
    print("=" * 70)
    print("MBT Prementalizing Modes Comparison")
    print("=" * 70)

    observed = 1  # sent_text
    print(f"\nScenario: Friend {action_labels[observed]}")

    print("\n--- Healthy Mentalizing ---")
    h = healthy_mentalizing()[:, observed]
    print(f"  P(doesnt_care)={h[0]:.3f}, P(neutral)={h[1]:.3f}, P(cares)={h[2]:.3f}")

    print("\n--- Psychic Equivalence ---")
    pe_ab = psychic_equiv_abandoned()[:, observed]
    pe_lov = psychic_equiv_loved()[:, observed]
    print(f"  Feeling abandoned: P(doesnt_care)={pe_ab[0]:.3f}, P(cares)={pe_ab[2]:.3f}")
    print(f"  Feeling loved:     P(doesnt_care)={pe_lov[0]:.3f}, P(cares)={pe_lov[2]:.3f}")

    print("\n--- Pretend Mode (no reality testing) ---")
    pm = pretend_mode()
    print(f"  P(doesnt_care)={pm[0]:.3f}, P(neutral)={pm[1]:.3f}, P(cares)={pm[2]:.3f}")
    print("  (Same regardless of friend's actual action)")

    print("\n--- Hypermentalizing ---")
    hm = hypermentalizing()[:, observed]
    print(f"  P(doesnt_care)={hm[0]:.3f}, P(neutral)={hm[1]:.3f}, P(cares)={hm[2]:.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.imshow(healthy_mentalizing(), cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(action_labels)))
    ax.set_xticklabels(action_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(mental_labels)))
    ax.set_yticklabels(mental_labels)
    ax.set_title('Healthy Mentalizing')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    x = np.arange(len(mental_labels))
    width = 0.25
    ax.bar(x - width, pe_ab, width, label='Feeling abandoned', color='coral')
    ax.bar(x, h, width, label='Healthy', color='green')
    ax.bar(x + width, pe_lov, width, label='Feeling loved', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(mental_labels)
    ax.set_title('Psychic Equivalence\n(same evidence, different feelings)')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    im = ax.imshow(hypermentalizing(), cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(action_labels)))
    ax.set_xticklabels(action_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(mental_labels)))
    ax.set_yticklabels(mental_labels)
    ax.set_title('Hypermentalizing')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    modes = ['Healthy', 'Psych Eq\n(aband.)', 'Pretend', 'Hyper']
    colors = ['green', 'coral', 'purple', 'orange']
    results = [h, pe_ab, pm, hm]
    width = 0.2
    for i, (r, m, c) in enumerate(zip(results, modes, colors)):
        ax.bar(x + (i-1.5)*width, r, width, label=m, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(mental_labels)
    ax.set_title(f'Mode Comparison (after "{action_labels[observed]}")')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('examples/mbt_modes_results.png', dpi=150)
    print("\nSaved to examples/mbt_modes_results.png")
