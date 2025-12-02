"""
BPD Abandonment Fear Scenario (from frontiers.tex Case Simulation 1)

Jane (BPD traits) sends Alex an urgent text. When Alex doesn't reply for an hour,
Jane's mind races with abandonment interpretations. Under high stress, she enters
"psychic equivalence" - her feeling of abandonment becomes unquestionable reality.
"""

import jax
import jax.numpy as np
from memo import memo

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Alex's possible intentions: 0=rejecting, 1=neutral/busy, 2=supportive
AlexIntention = [0, 1, 2]
intention_labels = ['rejecting', 'busy', 'supportive']

# Observable behaviors: 0=no reply, 1=delayed, 2=quick
ReplyBehavior = [0, 1, 2]
behavior_labels = ['no_reply_60min', 'delayed_reply', 'quick_reply']


@jax.jit
def bpd_prior(intention):
    """Jane's prior with abandonment fear schema."""
    weights = np.array([0.6, 0.3, 0.1])
    return weights[intention]


@jax.jit
def secure_prior(intention):
    """A securely attached person's prior."""
    weights = np.array([0.1, 0.4, 0.5])
    return weights[intention]


@jax.jit
def behavior_likelihood(behavior, intention):
    """P(behavior | intention)"""
    # rows = intention, cols = behavior
    matrix = np.array([
        [0.7, 0.2, 0.1],   # rejecting
        [0.3, 0.5, 0.2],   # busy
        [0.1, 0.3, 0.6],   # supportive
    ])
    return matrix[intention, behavior]


# =============================================================================
# Model 1: Jane's basic inference (BPD prior)
# =============================================================================

@memo
def jane_bpd_basic[i: AlexIntention, b: ReplyBehavior]():
    # Jane with BPD prior infers Alex's intention from reply behavior
    cast: [alex, jane]
    jane: thinks[
        alex: given(intention in AlexIntention, wpp=bpd_prior(intention)),
        alex: given(behavior in ReplyBehavior, wpp=behavior_likelihood(behavior, intention))
    ]
    jane: knows(i)
    jane: observes [alex.behavior] is b
    return jane[E[alex.intention == i]]


@memo
def observer_secure[i: AlexIntention, b: ReplyBehavior]():
    # A securely attached observer making the same inference
    cast: [alex, observer]
    observer: thinks[
        alex: given(intention in AlexIntention, wpp=secure_prior(intention)),
        alex: given(behavior in ReplyBehavior, wpp=behavior_likelihood(behavior, intention))
    ]
    observer: knows(i)
    observer: observes [alex.behavior] is b
    return observer[E[alex.intention == i]]


# =============================================================================
# Model 2: Psychic equivalence (prior only, no updating)
# =============================================================================

@memo
def psychic_equivalence[i: AlexIntention]():
    # Full psychic equivalence: Jane's prior is reality, no evidence use
    cast: [alex, jane]
    jane: thinks[
        alex: given(intention in AlexIntention, wpp=bpd_prior(intention))
    ]
    jane: knows(i)
    return jane[E[alex.intention == i]]


# =============================================================================
# Model 3: Stress-modulated (computed externally)
# =============================================================================

def jane_under_stress(observed_behavior, stress_level):
    """
    Jane's inference with stress-dependent mentalizing.
    Computed by blending posterior with prior based on stress.

    stress_level: 0=low (full updating), 1=moderate, 2=high (prior only)
    """
    # Evidence weight: how much to use posterior vs prior
    evidence_weights = np.array([1.0, 0.5, 0.1])
    evidence_weight = evidence_weights[int(stress_level)]

    # Get posterior from Bayesian inference
    posterior = jane_bpd_basic()[:, int(observed_behavior)]

    # Get prior (psychic equivalence)
    prior = psychic_equivalence()

    # Blend based on stress
    return evidence_weight * posterior + (1 - evidence_weight) * prior


if __name__ == "__main__":
    print("=" * 70)
    print("BPD Abandonment Fear Scenario")
    print("=" * 70)

    print("\n--- Basic inference matrices ---")
    print("\nJane (BPD prior) [intention x behavior]:")
    print(jane_bpd_basic())

    print("\nSecure observer [intention x behavior]:")
    print(observer_secure())

    no_reply = 0
    print(f"\n--- After observing: {behavior_labels[no_reply]} ---")
    print(f"Jane (BPD):  {jane_bpd_basic()[:, no_reply]}")
    print(f"Secure:      {observer_secure()[:, no_reply]}")

    print("\n--- Stress effects on Jane's inference ---")
    for stress in [0, 1, 2]:
        result = jane_under_stress(no_reply, stress)
        stress_label = ['low', 'moderate', 'high'][stress]
        print(f"  Stress={stress_label}: P(reject)={result[0]:.3f}, P(busy)={result[1]:.3f}, P(support)={result[2]:.3f}")

    print("\n--- Psychic equivalence (prior only) ---")
    pe = psychic_equivalence()
    print(f"  P(reject)={pe[0]:.3f}, P(busy)={pe[1]:.3f}, P(support)={pe[2]:.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    x = np.arange(len(intention_labels))
    width = 0.35
    bpd_post = jane_bpd_basic()[:, no_reply]
    secure_post = observer_secure()[:, no_reply]
    ax.bar(x - width/2, bpd_post, width, label='BPD (Jane)', color='coral')
    ax.bar(x + width/2, secure_post, width, label='Secure', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(intention_labels)
    ax.set_ylabel('P(intention | no_reply)')
    ax.set_title('Inference after no reply: BPD vs Secure')
    ax.legend()

    ax = axes[0, 1]
    stress_vals = [0, 1, 2]
    p_reject = [float(jane_under_stress(no_reply, s)[0]) for s in stress_vals]
    ax.bar(['low', 'moderate', 'high'], p_reject, color='coral')
    ax.set_ylabel('P(Alex is rejecting)')
    ax.set_xlabel('Stress level')
    ax.set_title('Stress -> Psychic equivalence')
    ax.set_ylim(0, 1)
    ax.axhline(y=float(bpd_prior(0)), color='gray', linestyle='--', label='BPD Prior')
    ax.axhline(y=float(secure_post[0]), color='steelblue', linestyle=':', label='Secure posterior')
    ax.legend()

    ax = axes[1, 0]
    im = ax.imshow(jane_bpd_basic(), cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(behavior_labels)))
    ax.set_xticklabels(behavior_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(intention_labels)))
    ax.set_yticklabels(intention_labels)
    ax.set_title('Jane (BPD) inference matrix')
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(observer_secure(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(behavior_labels)))
    ax.set_xticklabels(behavior_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(intention_labels)))
    ax.set_yticklabels(intention_labels)
    ax.set_title('Secure observer inference matrix')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('examples/bpd_abandonment_results.png', dpi=150)
    print("\nSaved to examples/bpd_abandonment_results.png")
