# Phase 1 — Peer Reviewer 3 (Perspective): contract paraphrase and scoring plan

Contract: `reviewer/reviewer_full/v1` (mode `reviewer_full`, panel 5). Paper metadata only: title "The Guarantee Holds and the Task Fails: Absorbing Stalls of Learned Policies under CBF Filters"; field safe reinforcement learning / shielding of learned policies, pre-registered simulation study with two propositions; about 3 400 words of main text, 10 PMLR pages including eight figures and one table, submitted to L4DC 2027. Reviewer identity (Card 4): safe-RL and shielding researcher, practice-oriented. No manuscript file has been opened before this file was written.

## Contract Paraphrase

**D1 — methodology_rigor (mandatory).** From where I sit, this dimension asks whether a safe-RL or shielding practitioner could take the study design at face value: a pre-registered simulation study has to state what was registered, what was measured, how seeds and episodes were paired, how the statistics were computed, and what a reader needs to reproduce the numbers. I am not the statistics reviewer; my contribution on D1 is whether the reproducibility affordances and the design are those that the safe-RL community would expect (seed counts, baselines a practitioner would actually try, archived artefacts), and whether the two propositions are stated with enough precision that the experiments can be said to test them.

**D2 — domain_accuracy (mandatory).** For me this dimension is whether the paper describes the adjacent literatures I know (shielding, safe exploration, training through a filter, reward shaping around filter interventions, recurrent and memory-based policies, discounting) accurately when it touches them, and whether its claims about what "learning" can or cannot do are consistent with what is known there. A mis-statement of a cited method, or a claim about learned policies that the safe-RL evidence base contradicts, counts against this dimension.

**D3 — argumentative_coherence (mandatory).** The thesis announced by the title is that a safety guarantee can hold while the task fails. The dimension asks whether the chain from the propositions through the experiments to the conclusions is internally consistent and whether each claim is sized to its evidence. From the safe-RL angle the specific hazard is over-reach: inferring from a bounded set of tested remedies to a general statement about what learning can remove, or from one game to a class of systems. I will judge whether the qualifiers are present where the evidence is partial.

**D4 — cross_disciplinary_relevance (high).** This is my home dimension. The paper sits between control (CBF filters) and learning (policies trained by RL). The dimension asks whether a reader from either side can follow the framing and definitions, whether the implications are stated in terms that a safe-RL practitioner, a formal-shielding researcher, or a roboticist deploying a filtered policy could act on, and whether any claim that reaches into another field (for example about liveness, exploration, or temporal abstraction) is substantiated rather than gestured at.

**D5 — writing_and_structure (normal).** For a 10-page PMLR paper with eight figures and one table, this dimension asks whether the space is well spent, whether the naming of experiments and policies is legible to someone who has never seen the authors' internal records, whether the figures can be read at their printed size, and whether the structure matches L4DC conventions. Secondary in priority but, with eight figures in ten pages, the figure legibility question is concrete and checkable.

## Scoring Plan

### D1: methodology_rigor
- dimension_id: D1
- what_to_look_for: an explicit statement of what was pre-registered and when relative to the data; seed and episode counts per comparison; a paired or otherwise matched evaluation design with the test named; a stated policy on failed hypotheses (reported, not hidden); a release repository or archived records that a reader could use; baselines that a safe-RL practitioner would consider natural (at minimum: penalty shaping, observation augmentation, longer horizon or discount, training with the filter in the loop); propositions stated with explicit assumptions.
- what_triggers_block: a central numerical claim with no stated N, no test, or no record; the pre-registration claim unsupported by any artefact; the study design unable in principle to distinguish the paper's explanation from an obvious alternative (for example, the failure being an artefact of one seed or one opponent) with no acknowledgement.
- what_triggers_warn: pre-registration asserted but the registered-versus-exploratory boundary not drawn for the reader; a natural safe-RL baseline missing without justification; seed counts low enough that per-family conclusions rest on two or three runs and the text does not qualify them; reproducibility materials referenced but not described well enough to locate.

### D2: domain_accuracy
- dimension_id: D2
- what_to_look_for: accurate characterisation of shielding and safe-RL prior work where cited; correct use of terms such as shield, filter, forward invariance, deadlock, liveness, memoryless/Markov policy, discount horizon; claims about learned policies (what training does or cannot do) that match what the safe-RL literature has shown; no statement that a cited method does something it does not.
- what_triggers_block: a factual mis-statement about a cited method or result that the argument depends on; a domain term used in a way that makes a claim false (for example calling something a guarantee that the filter does not provide); a claim that learning in general cannot do X when a standard, named technique does X and is neither tested nor discussed.
- what_triggers_warn: a terminological looseness that could mislead an adjacent reader but does not change the result; a prior-work description that is one-sided or omits the part most relevant to the paper's claim; a claim about learned policies stated more strongly than the tested policy class justifies, but with a qualifier somewhere in the paper.

### D3: argumentative_coherence
- dimension_id: D3
- what_to_look_for: a one-sentence central claim recoverable from the abstract; each proposition connected to at least one experiment that bears on it; conclusions whose scope matches the experimental scope (policy class, memory, horizon, dynamics, number of filters); the negative results (remedies that did not work) stated with the same prominence as the positive result; the title's claim earned by the evidence rather than by framing.
- what_triggers_block: the central claim depends on an inference the experiments do not support (for example, "learning cannot remove it" when memory or horizon were not varied and no argument covers them); a contradiction between the propositions and the experimental finding that is not addressed; a remedy presented as the contribution without a comparison that shows it is needed.
- what_triggers_warn: generalisation language (cannot, always, any learned policy) that outruns the tested conditions but is softened elsewhere; a dissociation or side result that does not feed the central argument and is not marked as such; an argument whose load-bearing step is a bound or heuristic presented without its assumptions.

### D4: cross_disciplinary_relevance
- dimension_id: D4
- what_to_look_for: definitions given in the paper's own terms before use, so that a learning reader can follow the control content and a control reader the learning content; implications stated for the people who would act on them (someone training a policy behind a filter, someone designing the filter, someone deploying); connections to adjacent ideas that bear directly on the mechanism (liveness in formal shielding, options or temporal abstraction, exploration under uninformative inputs, discount-horizon effects) either made or at least not contradicted; interdisciplinary claims (for example about formal methods or about human oversight) supported by citation or evidence.
- what_triggers_block: the framing is unintelligible to one of the two core readerships (a learning reader cannot tell what the filter does; a control reader cannot tell what the policy sees or optimises), or an interdisciplinary claim is made that is unsupported and central.
- what_triggers_warn: definitions present but scattered or introduced after first use; implications stated only for one readership; an obvious adjacent connection that would change how a practitioner reads the result is absent; a claim about practice (adoption, cost, when to use the remedy) made without the conditions under which it holds.

### D5: writing_and_structure
- dimension_id: D5
- what_to_look_for: experiment and policy names that carry meaning without a key, or a key given once and early; figures legible at their printed width (axis labels, legends, line distinguishability, text size at roughly 7 to 12 cm wide); a table that is readable within the column; sections proportioned to their contribution; L4DC/PMLR conventions respected (page count, abstract, references).
- what_triggers_block: figures that cannot be read at printed size for the results that carry the main claim, with no table or text restating them; naming so opaque that the results cannot be followed without external records.
- what_triggers_warn: some figures too small or too dense at printed width; names that need a lookup table the reader must keep in mind; a section that takes space disproportionate to its role in a ten-page paper; small convention slips.

[CONTRACT-ACKNOWLEDGED]
