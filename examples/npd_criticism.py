"""
NPD Criticism Interpretation Scenario (from frontiers.tex Case Simulation 2)

John (NPD traits) presents a project. A colleague makes a mild critical remark.
John immediately perceives it as a personal attack, attributing it to jealousy
rather than valid feedback. His grandiose self-image acts as a strong prior.
"""

import jax
import jax.numpy as np
from memo import memo

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Interpretations of colleague's critique
# 0=valid criticism, 1=minor issue exaggerated, 2=envy/malice
Interpretation = [0, 1, 2]
interpretation_labels = ['valid_criticism', 'minor_exaggeration', 'envy_malice']

# Observable critique: 0=none, 1=mild, 2=strong
CritiqueBehavior = [0, 1, 2]
critique_labels = ['no_critique', 'mild_critique', 'strong_critique']


@jax.jit
def npd_prior(interpretation):
    """John's prior with grandiose self-image."""
    weights = np.array([0.05, 0.20, 0.75])
    return weights[interpretation]


@jax.jit
def realistic_prior(interpretation):
    """A realistic person's prior."""
    weights = np.array([0.40, 0.35, 0.25])
    return weights[interpretation]


@jax.jit
def critique_likelihood(critique, interpretation):
    """P(critique | interpretation)"""
    # rows = interpretation, cols = critique
    matrix = np.array([
        [0.1, 0.5, 0.4],   # valid: likely to voice
        [0.3, 0.5, 0.2],   # minor: might mention
        [0.2, 0.5, 0.3],   # envy: would criticize
    ])
    return matrix[interpretation, critique]


@memo
def john_npd_inference[i: Interpretation, c: CritiqueBehavior]():
    # John (NPD) interprets critique with grandiose bias
    cast: [colleague, john]
    john: thinks[
        colleague: given(interpretation in Interpretation, wpp=npd_prior(interpretation)),
        colleague: given(critique in CritiqueBehavior, wpp=critique_likelihood(critique, interpretation))
    ]
    john: knows(i)
    john: observes [colleague.critique] is c
    return john[E[colleague.interpretation == i]]


@memo
def realistic_inference[i: Interpretation, c: CritiqueBehavior]():
    # Realistic person interprets the same critique
    cast: [colleague, observer]
    observer: thinks[
        colleague: given(interpretation in Interpretation, wpp=realistic_prior(interpretation)),
        colleague: given(critique in CritiqueBehavior, wpp=critique_likelihood(critique, interpretation))
    ]
    observer: knows(i)
    observer: observes [colleague.critique] is c
    return observer[E[colleague.interpretation == i]]


if __name__ == "__main__":
    print("=" * 70)
    print("NPD Criticism Interpretation Scenario")
    print("=" * 70)

    print("\n--- Inference matrices ---")
    print("\nJohn (NPD) [interpretation x critique]:")
    print(john_npd_inference())

    print("\nRealistic observer [interpretation x critique]:")
    print(realistic_inference())

    mild = 1
    print(f"\n--- After observing: {critique_labels[mild]} ---")
    npd_r = john_npd_inference()[:, mild]
    real_r = realistic_inference()[:, mild]

    print(f"John (NPD):  P(valid)={npd_r[0]:.3f}, P(minor)={npd_r[1]:.3f}, P(envy)={npd_r[2]:.3f}")
    print(f"Realistic:   P(valid)={real_r[0]:.3f}, P(minor)={real_r[1]:.3f}, P(envy)={real_r[2]:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    x = np.arange(len(interpretation_labels))
    width = 0.35
    ax.bar(x - width/2, npd_r, width, label='NPD (John)', color='coral')
    ax.bar(x + width/2, real_r, width, label='Realistic', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(interpretation_labels, rotation=15)
    ax.set_ylabel('P(interpretation | mild_critique)')
    ax.set_title('Inference after mild critique')
    ax.legend()

    ax = axes[1]
    im = ax.imshow(john_npd_inference(), cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(critique_labels)))
    ax.set_xticklabels(critique_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(interpretation_labels)))
    ax.set_yticklabels(interpretation_labels)
    ax.set_title('John (NPD) matrix')
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(realistic_inference(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(critique_labels)))
    ax.set_xticklabels(critique_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(interpretation_labels)))
    ax.set_yticklabels(interpretation_labels)
    ax.set_title('Realistic observer matrix')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('examples/npd_criticism_results.png', dpi=150)
    print("\nSaved to examples/npd_criticism_results.png")
