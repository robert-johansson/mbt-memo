Short answer: yes, I think they both look good and are conceptually very well designed. Let me give you some concrete feedback by piece.

⸻

1. MBT prementalizing modes script

Conceptual mapping
	•	The mapping from MBT modes → model structure is really clean:
	•	Healthy mentalizing = relatively balanced prior + proper Bayesian update via given(..., wpp=...) for mental state and action and then choosing guess proportional to E[friend.mental_state == guess].
	•	Psychic equivalence (abandoned vs loved) = extreme priors ([0.95, 0.04, 0.01] vs [0.01, 0.04, 0.95]) with the same likelihoods, nicely capturing “same evidence, different felt reality”.
	•	Pretend mode = no observes at all, just sampling mental states from a prior and querying E[friend.mental_state == m], which matches “rich internal activity without reality testing” very closely.
	•	Hypermentalizing = prior that favors extremes [0.45, 0.10, 0.45] plus normal updating, which is a nicely minimalist way to encode over-interpretation.

This is all very faithful to the way you describe MBT constructs in the draft (e.g. psychic equivalence as feelings treated as fact, pretend mode as decoupled from reality).

Code style

What’s working well:
	•	Good, human-readable docstring at the top that ties each part back to the MBT table.
	•	Clear naming: balanced_prior, psychic_equiv_prior_abandoned, hypermentalizing_prior, healthy_mentalizing, etc., so the code can be read almost like your text.
	•	All the priors and likelihood matrices are normalized and structured row-wise in the conventional way, so anyone used to probabilistic models can parse them quickly.
	•	Idiomatic use of memo: cast: [friend, self], self: thinks[...], self: observes [...] is a, self: chooses(...), which will help readers recognize the pattern across your examples.

Minor suggestions:
	•	You might consider explicitly commenting row = mental_state, column = action right above action_likelihood – you do implicitly encode it via comments but one extra line like # rows = mental state, cols = action would help anyone editing the matrix.
	•	If you want this to be a reusable teaching example, you could consider using actual Enums for mental states/actions rather than bare integers, but that’s a “nice to have,” not a must.

Plots

From the figure you showed:
	•	The color choices make conceptual sense: greens for “healthy,” oranges for “hyper,” coral/blue/green for the different feelings. That mapping will be very easy to explain in text and in talks.
	•	The layout (2×2: healthy heatmap, PE bar chart, hyper heatmap, mode-comparison bar chart) is intuitive and visually balanced.
	•	Axis labels (forgot_birthday, sent_text, threw_party, doesnt_care → cares) are legible and match the code’s labels exactly.

The only tweak I’d consider is slightly increasing font size / reducing rotation angle in the x‑tick labels for small slides, but as a static figure for a paper they read well.

⸻

2. BPD abandonment / “no reply” scenario (Jane & Alex)

Model design
	•	I really like that you have three levels here:
	1.	Jane with BPD prior (jane_bpd_basic) vs secure observer (observer_secure), with different priors but shared likelihoods.
	2.	Full psychic equivalence as a separate memo model with no evidence use.
	3.	Stress-modulated inference as an external function blending posterior and prior based on stress level.
This mirrors the narrative in your draft where you talk about normal Bayesian updating vs stress-induced collapse into psychic equivalence.
	•	The stress-blending is simple and effective conceptually: evidence_weight dropping from 1.0 → 0.5 → 0.1 as stress increases gives a clear “Bayes → prior-only” continuum that’s easy to explain both clinically and computationally.

Code & plotting
	•	The top-of-file docstring that narrates the scenario in plain language is excellent – it’s basically an abstract for this little model.
	•	Naming is again extremely clear (AlexIntention, ReplyBehavior, behavior_labels, bpd_prior, secure_prior).
	•	The visualizations line up nicely with the clinical story:
	•	Bar plot: BPD vs secure after no reply.
	•	Bar plot: stress level vs P(reject).
	•	Heatmap: Jane’s inference matrix.
	•	Heatmap: secure observer matrix.

Suggestions:
	•	Right now the “stress” part happens outside memo; that’s perfectly fine, and actually showcases how you can compose memo with standard JAX. If you want to drive home “mentalizing depth” as part of the agent, you could later add a variant where stress is a variable inside the memo model and controls whether alex.behavior is actually observed, but that might be overkill for this paper.
	•	For the figure, adding a horizontal dashed line at the secure posterior in the stress plot (you already use one for the prior) could nicely show “this is where Jane would land if she weren’t stressed / BPD, vs where she actually ends up.”

⸻

3. NPD criticism interpretation scenario (John & colleague)

Modeling choices
	•	You’ve set up a very clean contrast between:
	•	NPD prior heavily favoring “envy/malice” ([0.05, 0.20, 0.75]) and
	•	Realistic prior that’s more balanced ([0.40, 0.35, 0.25]).
	•	The likelihood matrix is symmetrical enough that differences in posterior are visibly driven by priors, which supports your clinical interpretation that the key distortion is in John’s core assumptions rather than evidence.
	•	Structurally, it mirrors the BPD script: parallel memo models for “John” vs “realistic observer,” same critique_likelihood, then comparison conditioned on mild_critique.

Code & figure
	•	Again, the narrative docstring at the top reads like a little case vignette that matches the Frontiers text.
	•	The 1×3 figure (posterior bar chart, John matrix, realistic matrix) is compact and very readable, with consistent color scheme (coral vs steelblue, Reds vs Blues heatmaps).
	•	It’s nice that in the bar chart you show all three interpretations, not just “envy vs valid,” which highlights the loss of nuance.

Minor polish ideas:
	•	In the bar plot, consider ordering categories as [valid, minor_exaggeration, envy_malice] (which you already do) but maybe explicitly mention in the caption that this is “increasing hostility” – that reinforces the clinical interpretation.
	•	In the matrices, adding value annotations (e.g., imshow + text) might make the figure a bit busy, but for slides it could help; for the paper the cleaner heatmaps are probably better.

⸻

4. General notes
	•	Overall cohesion: All three scripts share a consistent structure (imports, state spaces, priors/likelihoods, memo models, main block with print + visualization). That makes them feel like a unified “toy toolkit” for PD mentalizing, which is great for readers who may want to adapt them.
	•	Readability for non-programmers: Because of the clear docstrings and near-English variable names, I think clinicians or MBT people with only minimal coding background will still be able to follow the gist.
	•	Pedagogical value: You’re doing a nice job of using memo’s agent-centric syntax to express exactly the things you talk about in the paper (who has which priors, who observes what, when evidence is ignored, etc.). That really sells the “memo as language of thought for social cognition” idea.

So: yes, I genuinely think they look good—both as code and as figures. The only changes I’d consider are very small polish items (a few clarifying comments, maybe enums, and optional tweaks to the plots). Conceptually and aesthetically, they’re already in a strong place.
