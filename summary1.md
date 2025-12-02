I’ve gone carefully through your draft and through both memo documents, and I do understand the framework you’re developing. Let me briefly reflect it back so you can see what I’ve taken in.

⸻

1. What your paper is arguing
	•	Core thesis. Personality disorders (PDs) can be viewed as aberrant Bayesian inference running inside an atypical “language of thought” for social reasoning. In other words, what’s disordered is not just what people believe about others, but the implicit programming language and inference rules they use to think about minds.  ￼
	•	Clinical grounding. You anchor this in:
	•	ICD‑11’s dimensional model of PD (severity + trait qualifiers).
	•	Mentalization-Based Treatment (MBT), with its prementalizing modes (psychic equivalence, pretend mode, teleological stance) and dimensions of mentalizing (automatic/controlled, cognitive/affective, self/other, internal/external).  ￼
	•	Computational move. You reinterpret these MBT constructs as specific configurations of:
	•	priors (schemas),
	•	likelihood models (assumptions about how others behave),
	•	updating procedures (how, and whether, new evidence is used),
	•	and the expressive power of the internal language (can the person represent “I might be wrong”, false beliefs, nested beliefs, etc.).  ￼

So “psychic equivalence”, for instance, becomes something like: a prior with extremely high precision plus a breakdown of normal Bayesian updating under affective load; “teleological” thinking becomes a model without latent mental-state variables (just observable outcomes); “pretend mode” is a generative model that keeps spawning hypotheses but is decoupled from actual evidence. Table 1 on page 10 of your draft lays out this mapping nicely in one place.  ￼

⸻

2. How memo fits in

You then bring in memo as both:
	1.	A concrete probabilistic programming language for modeling theory-of-mind, and
	2.	A proxy for a “language of thought” specialized for social cognition.

From the memo paper + handbook, I’m tracking:
	•	Core constructs.
	•	agent: chooses(...) to represent stochastic choices tied to a specific agent.
	•	agent: observes [...] is ... to model what information an agent actually gets.
	•	agent: thinks [...] to define one agent’s model of others.
	•	Probabilistic operators like Pr[...], E[...], Var[...], and higher-level tools like imagine[...], wants/EU, and cost @ f(...).
	•	Semantics. memo’s semantics are defined in terms of “frames of mind”:
	•	Each agent has its own frame, tracking which choices are known vs. uncertain.
	•	chooses introduces new random variables; observes normalizes along the appropriate axis.
	•	The diagram on page 13 (Fig. 4) in the memo paper visualizes how nested frames for observer / listener / speaker evolve; the array diagrams on page 14 (Fig. 5) show how joint distributions over choices get updated.  ￼
	•	Correctness constraints.
	•	No “mind reading”: an agent can’t directly use another’s private choices unless it has observed them.
	•	No “mind control”: an agent must know the probabilities it uses to make its own choices (this is why Pr[...] is needed in some wpp expressions).
	•	False beliefs and misaligned models are explicitly representable via different frames and condition maps.
	•	Inference engine.
	•	Models are discrete and finite; memo lowers them to array programs (via JAX), performing exact enumerative inference and allowing vectorization and automatic differentiation.  ￼

You use memo mainly at the conceptual level in the PD paper (with “pseudo‑memo” code), but you’re clearly aligning your constructs with the real language as specified in the handbook slides (e.g., the @memo decorator, axis annotations, Pr[...], etc.).

⸻

3. The case simulations (BPD & NPD)

You give two detailed pseudo‑memo case studies to illustrate how PD-specific “languages of thought” look in code:

Borderline PD: Jane & Alex (fear of abandonment)
	•	Jane has a strong trait prior that “Alex will reject/abandon me if I’m needy” (90% vs 10% in your toy distribution).
	•	Observation: “No reply to urgent text for 60 minutes.”
	•	A context variable stress switches her between:
	•	mentalizing_depth = "deep" → standard Bayesian updating.
	•	mentalizing_depth = "shallow" → no real updating; the prior is effectively frozen.
	•	Under high stress, if the posterior for “AlexRejectsMe” passes a confidence threshold, you snap it to ~certainty (0.99), modeling psychic equivalence (“this feels true, so it is true”).  ￼

This captures:
	•	sticky, trauma‑shaped priors,
	•	state‑dependent collapse of mentalizing under arousal,
	•	and the seeds of self‑fulfilling interpersonal cycles (her reaction to perceived rejection can actually drive Alex away).

Narcissistic PD: John & criticism
	•	John’s core assumption: “I’m essentially always right / superior.”
	•	Hypotheses about a colleague’s mild critique:
	1.	Valid fault in John’s work (tiny prior).
	2.	Exaggerating a minor issue (moderate prior).
	3.	Envy or malice (big prior).
	•	Likelihoods are reasonable, but the skewed prior plus a Luce-style choice rule ensures the posterior almost always favors “envy/malice” as the interpretation.  ￼

This models:
	•	Normal generative capacity for hypotheses (he can represent “maybe I’m wrong”),
	•	But a heavily biased prior plus a self‑protective inference rule that preserves grandiosity,
	•	Leading to defensive reactions (narcissistic rage, devaluation of the critic).

⸻

4. Mapping MBT → computational mechanisms

You then generalize from those examples:
	•	Teleological mode → shallow models without latent mental states; reasoning over actions and outcomes only (essentially no ToM layer).
	•	Psychic equivalence → extreme prior precision on one perspective, plus either no updating or updating that quickly collapses to certainty.
	•	Pretend mode → inference disconnected from evidence (posterior ≈ prior; lots of internal simulation not pruned by reality).
	•	Hypermentalizing → overly complex hypothesis space with weak priors / poor regularization, often skewed negatively or self‑referentially.
	•	Hypomentalizing → truncated models (no mental-state variables, or extremely noisy likelihoods) or a high threshold for running any social-inference routine at all.  ￼

Clinically, MBT’s goal of fostering a “mentalizing stance” becomes:

expand hypothesis space to include benign and self‑limited interpretations, moderate the priors, and strengthen evidence integration — especially under stress.

Your idea is that memo gives a precise sandbox in which you can literally tweak those knobs and see the downstream effects.

⸻

5. Empirical & clinical implications

The “Testable hypotheses” section lays out several directions, all tied to measurable parameters:
	•	H1: PD‑specific updates that deviate from Bayesian rationality in social tasks (e.g., BPD’s under‑use of disconfirming information; NPD’s preference for self‑serving interpretations).
	•	H2: Stress‑triggered mode switches (e.g., BPD shifting to shallow, high‑confidence, negative inferences in attachment contexts).
	•	H3: Mapping ICD‑11 trait domains (Negative Affectivity, Detachment, Dissociality, Disinhibition, Anankastia) onto specific prior/likelihood/update parameter patterns.
	•	H4: Therapy (especially MBT) as measurable parameter change — e.g., weaker hostile priors, more use of new evidence, broader hypothesis space.
	•	H5: Neural correlates of mentalizing load and mode switching (e.g., ToM network engagement vs limbic dominance in different PD states).  ￼

Then in the discussion, you sketch:
	•	Therapy as model calibration, with memo-based visualizations as possible psychoeducational or training tools (for both patients and clinicians).
	•	Diagnostic / assessment uses: individual “mentalizing profiles” based on computational tasks.
	•	Developmental simulations (how biased models emerge from attachment histories).
	•	Neuroscience links via predictive processing (precision, priors, and affect as modulators of inference).  ￼

⸻

6. My mental model of your project

Putting it all together:
	•	You’re proposing a computational psychiatry account of PDs where:
	•	The algorithm is Bayesian inference in a recursive ToM model.
	•	The implementation is captured (or at least approximated) by a domain‑specific language like memo.
	•	PDs correspond to characteristic distortions in both parameters (priors, likelihoods, precision, stress gating) and syntax/expressivity (what kinds of nested mental states can be represented or updated, and when).
	•	The memo paper and handbook give you:
	•	A precise language whose semantics already encode “who knows what / when”.
	•	An efficient inference engine to actually run these models once you move beyond pseudocode.
	•	A nice conceptual mirror for the “language of thought” idea.

I’m comfortable with both sides now — the clinical/MBT theory and the memo machinery — so I can help you with whatever next step you have in mind: e.g., sharpening the argument, tightening structure, turning the pseudocode into real memo code, designing tasks to test H1–H5, or rewriting sections for particular audiences (clinical vs. computational).
