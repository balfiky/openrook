# Research review for humanlike autonomy in Nūr

Research cutoff: 18 October 2026. This review informs the [implementation and evaluation plan](experience_grounded_autonomy_plan.md). It does not change the running agent or authorize a new study.

The strongest direction is an agent whose choices, predictions, skills, and relationships change through consequences of its own activity. Persistent emotion labels and autobiographical prose are not sufficient evidence of that development. Recent work also challenges the assumption that a capable language model already supplies humanlike goal selection if given a sufficiently good prompt.

Three objectives must remain separate: building a functioning self-directed agent, matching particular human behavioral patterns, and investigating subjective experience. The first two admit useful experiments; the research reviewed here supplies no validated engineering recipe or benchmark for the third. Better task performance is not necessarily closer human resemblance.

## Findings that change the approach

1. **Study goal choice, not only task execution.** Human experiments and recent model comparisons provide controlled tasks where participants choose what to learn. These are better starting points than an empty workspace and a long timer.
2. **Make experience predictive.** Retained information should improve later choices on unfamiliar instances, not merely appear in the agent's self-description.
3. **Treat emotion as a functional hypothesis.** Test whether goal-relative appraisal changes attention, learning, or decisions. Expressive language is a separate outcome.
4. **Measure trajectories, not just averages.** Within-agent persistence and recovery, between-agent variation, and transfer matter. Stable questionnaires or changing database values alone do not establish those properties.
5. **Borrow mechanisms selectively.** Active inference, intrinsic motivation, and world models offer useful hypotheses. None removes the need to declare initial preferences, environmental assumptions, and an evaluation target.

These are design conclusions from the combined evidence, not claims that any one paper establishes a complete architecture. The following ledger distinguishes human evidence, implemented systems, model evaluations, and proposals. Publication dates and revisions are distinguished where relevant; a recent upload is not automatically a recent experiment.

## Human behavior and motivation

### H1 Autonomy is not the absence of relationships

[Ryan and Deci, Self-Regulation and the Problem of Human Autonomy](https://selfdeterminationtheory.org/SDT/documents/2006_RyanDeci_Self-RegulationProblemofHumanAutonomy.pdf). Journal of Personality, December 2006; foundational theoretical synthesis.

Self-determination theory distinguishes self-endorsed regulation from independence or isolation. Cooperation can be autonomous; opposition can be externally driven. This is one psychological framework, not a definition of machine consciousness.

**Design implication:** do not reward Nūr for disagreeing with its owner, avoiding contact, or producing more actions. Evaluate continuity of choice and responsiveness to evidence, while allowing cooperation, rest, and changing one's mind.

### H2 Motivation involves distinguishable needs

[Howard, Slemp, and Wang, Need Support and Need Thwarting](https://journals.sagepub.com/doi/10.1177/01461672231225364). Personality and Social Psychology Bulletin, 2025 issue; first published online 30 January 2024. Meta-analysis of 637 student samples, 388,912 participants.

The study examines supportive and thwarting behavior around autonomy, competence, and relatedness. Its educational, largely correlational evidence does not supply universal machine reward coefficients.

**Design implication:** consider agency, competence, and relationships separately, but do not turn their names into arbitrary meters and call the result validated psychology. Any implemented need must have an explicit observable consequence and a competing baseline.

### H3 Human goal selection can track latent learning progress

[Molinaro and colleagues, Latent Learning Progress Drives Autonomous Goal Selection in Human Reinforcement Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12676560/). NeurIPS 2024; author manuscript deposited in PMC in December 2025. Behavioral experiment and computational modeling, 175 participants.

Participants selected learning goals in a structured task. Results support a contribution from inferred progress, including eliminating possible solutions before overt success improves. Individuals differed, partly with recognition of hierarchical structure. This is task-specific evidence, not a universal curiosity law.

**Design implication:** distinguish successful outcomes, reduced uncertainty, and verified progress through a search space. A failed attempt may teach something; repeating the same failure without new information need not.

### H4 Goal persistence is not explained by one drive

[Molinaro, Lidayan, and Collins, Investigating the Role of Task Representation Switch Costs in Goal Persistence](https://www.microsoft.com/en-us/research/publication/investigating-the-role-of-task-representation-switch-costs-in-goal-persistence/). CogSci, July 2026; two experiments, 125 participants.

Changing stimulus-action representations influenced persistence and the choice of a replacement goal, but explained only a small part of persistence. Human continuation is not simply optimal utility maximization or a fixed persistence threshold.

**Design implication:** record switching costs and alternatives. Test continuation after changed conditions instead of treating either relentless persistence or immediate abandonment as inherently humanlike.

### H5 Experience can simplify goal evaluation

[Molinaro and Collins, Reward Function Compression Facilitates Goal-Dependent Reinforcement Learning](https://www.nature.com/articles/s41467-026-78295-1). Nature Communications, 8 October 2026; peer-reviewed accepted manuscript, ahead of the final version of record.

Across six human experiments, learning was harder with larger goal spaces and improved when goal information admitted compression. Faster reward processing related to better learning. The proposed transition toward compact long-term rules is not an established neural implementation; algorithmic details remain open.

**Design implication:** test whether evidence-backed reusable rules reduce repeated deliberation and transfer to new instances. This motivates a procedural-memory experiment, not a claim that saving summaries reproduces human consolidation.

### H6 Personality development is usually not an abrupt rewrite

[Bühler and colleagues, Life Events and Personality Change](https://journals.sagepub.com/doi/10.1177/08902070231190219). European Journal of Personality, 2024 issue; online 23 July 2023. Preregistered meta-analysis of 44 studies, 89 samples, 121,187 participants.

Associations were generally small and dependent on event, trait, and measurement timing. Observational longitudinal associations do not establish a universal causal event-to-trait mapping.

**Design implication:** distinguish temporary state, learned expectation, and longer-lasting disposition. Require repeated behavioral measurements; do not script a permanent personality change after one insult, success, or autobiographical paragraph.

## Recent evidence about simulated people

### A1 Language models do not automatically select goals like people

[Molinaro and colleagues, Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task](https://arxiv.org/abs/2603.03295v2). Empirical paper; consulted arXiv revision dated 13 May 2026.

Five models were compared with human behavior in a controlled learning task. Models often concentrated on a discovered solution or learned poorly, with less within-model diversity; reasoning and persona steering offered limited improvements. Full-history access and the constrained task limit generalization. The result does not show that every model or engineered agent must fail.

**Design implication:** measure the backend alone before attributing behavior to Nūr. Separate freely selected practice goals from later tests covering neglected and transferable knowledge. A high success rate on repeatedly chosen easy goals can conceal weak learning.

### A2 Plausible averages can hide incorrect psychological dynamics

[Vaid and Whillans, Benchmarking Psychological Dynamics in Generative Agents](https://arxiv.org/abs/2610.04246v1). Preprint, 3 October 2026.

The benchmark tests 36 models using synthetic personas and six simulated days of workplace narratives. Broad relationships were easier to reproduce than within-person covariance and distributional structure. These are prompted state ratings against published human reference patterns, not an independently observed longitudinal population of autonomous agents.

**Design implication:** measure event-conditioned trajectories, variability, and recovery separately from mean state values. Matching a synthetic benchmark's aggregate correlations is not sufficient validation of Nūr's internal mechanism.

### A3 Persona change is not yet convincing developmental fidelity

[Wang and colleagues, Do AI Personas Grow?](https://arxiv.org/abs/2608.06485v1). Preprint, 6 August 2026; BFI-Adapt benchmark covering 14 models and 11 life events.

Event-conditioned trait shifts occurred, but were weakly event-specific and showed compressed variation relative to human reference ranges. Agreement with scenario-based choices was limited and model-dependent. Persistence checks used intervening dialogue, not human-scale developmental time.

**Design implication:** use no-event, paraphrase, delayed-test, and behavioral controls. Do not force Big Five changes to reproduce a desired trajectory or equate a changed questionnaire answer with durable character development.

### A4 Reliable self-description may not predict behavior

[Contreras, An LLM-Native Psychometric Instrument Reveals a Self-Report–Behavior Gap Across 25 Models](https://arxiv.org/abs/2606.09843v4). Preprint, consulted revision dated 7 October 2026.

The study combines repeated questionnaires with behavioral samples and human ratings. Reliable self-report structure did not translate into strong overall agreement with human-rated behavior. Agreement between an acting model and another model's ratings is not an adequate substitute.

**Design implication:** treat Nūr's diary, trait statements, and emotional descriptions as reports. Ground primary outcomes in actions and consequences; use blinded human ratings only where interpretation genuinely requires them.

### A5 Psychometrics still has a bounded role

[A Psychometric Framework for Evaluating and Shaping Personality Traits in Large Language Models](https://www.nature.com/articles/s42256-025-01115-6). Nature Machine Intelligence, 18 December 2025; peer-reviewed study of 18 models.

The work finds useful psychometric properties under particular model and prompting conditions and demonstrates steerability of expressed traits. That does not establish spontaneous personality development. Its controlled measurement setting differs from the broader behavioral tests above.

**Design implication:** retain questionnaires as secondary diagnostics with retest and wording controls. Neither dismiss every personality measure nor use one as the principal proof of independence.

### A6 Human behavior data can improve prediction

[Binz and colleagues, A Foundation Model to Predict and Capture Human Cognition](https://www.nature.com/articles/s41586-025-09215-4). Nature, 2 July 2025; Centaur.

Training on human experimental behavior improves predictions across cognitive tasks. A predictor fitted to human choices is a different product from an agent that develops its own goals through interaction.

**Design implication:** human trajectories are a plausible calibration or training resource if behavioral fidelity is the aim. Centaur is a comparator, not an assumed replacement brain for Nūr.

### A7 Predictive accuracy is not free-running validity

[Namazova and colleagues, Not Yet AlphaFold for the Mind](https://arxiv.org/abs/2508.07887). Preprint, 11 August 2025; evaluation of Centaur as a synthetic participant.

The authors report that accurate behavioral prediction can coexist with generated behavior that diverges systematically from humans. This counterevidence narrows the interpretation of prediction benchmarks rather than proving human-data training useless.

**Design implication:** evaluate trajectories generated by the agent's own previous choices, not only isolated predictions conditioned on human histories.

### A8 Individual simulation benefits from personal grounding

[LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals](https://arxiv.org/abs/2411.10109v3). Consulted revision dated 28 June 2026; original preprint November 2024; 1,052 participants.

Interview and survey grounding improves prediction of held-out responses. Reported GSS accuracies are normalized to participants' own retest consistency, not percentages of overall human likeness. This evaluates simulation of existing people, not autonomous formation of a new identity.

**Design implication:** a human digital twin and an experience-developing synthetic individual require different initialization and evaluation. Do not secretly seed the latter with a copied biography to make it appear more human.

## Implemented mechanisms worth testing

### M1 Goal-relative computational appraisal

[Zhang, Broekens, and Jokinen, Modeling Cognitive-Affective Processes With Appraisal and Reinforcement Learning](https://doi.org/10.1109/TAFFC.2024.3470555). IEEE Transactions on Affective Computing, April–June 2025 issue; online 27 September 2024.

The model formalizes appraisal checks using temporal-difference learning and evaluates predictions of human emotions in vignette studies. It connects events to goals, expectations, and coping possibilities rather than assigning emotion solely by event label. Validation is bounded to the studied tasks and measures.

**Design implication:** compare Nūr's fixed updates with outcome-grounded appraisal. Do not label a heuristic implementation of selected ideas as a reproduction of the full model.

### M2 Need-linked attention can have measurable consequences

[Grimbly and colleagues, Interoceptive Attention as Dynamic Homeostatic Prioritization in a Foraging Agent](https://arxiv.org/html/2608.04232v1). 4 August 2026; authors report SAB 2026 acceptance, with additional unreviewed supplementary analyses.

A hand-specified need-aligned allocation rule improves survival in a small gridworld at matched sensing budget; reversed and pathway controls test parts of the mechanism. It does not demonstrate human emotion or a learned prioritization rule. The main text and supplement describe different policy precision values; the supplement also specifies the executed objective without state-information gain. Resolve those implementation details before replication.

**Design implication:** test competing resource demands and causal pathways, not decorative emotional labels. A lightweight Nūr experiment need not adopt active inference or invent hunger and suffering.

### M3 Useful memory includes procedures and changing premises

[LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues](https://arxiv.org/abs/2605.12493v1). Preprint, 12 May 2026; 451 curated questions over web-agent histories.

The benchmark extends beyond factual recall to changing state, workflows, pitfalls, and premise awareness. The evaluated memory systems improve relevant retrieval, but answering questions about experience is not equivalent to autonomously using that experience successfully.

**Design implication:** retrieve goal-relevant prior attempts and conditional procedures, including contradictory or obsolete outcomes. Test subsequent execution rather than optimizing memory answers alone.

### M4 Learning from context is measurable but uneven

[Wang and colleagues, How Far Can LLMs Improve from Experience?](https://aclanthology.org/2025.emnlp-main.1304/). EMNLP 2025; peer-reviewed test-time learning study with a small eight-person human comparison.

Repeated semantic-game experience produced model improvement, but learning was less stable and efficient than in the human comparison. Task choice and the small comparator limit generalization.

**Design implication:** compare experienced and fresh agents on new instances at matched budgets. External memory can support genuine measured adaptation without implying that the hosted model's weights changed.

### M5 World-model learning is demonstrated in control tasks

[Hafner and colleagues, Mastering Diverse Control Tasks Through World Models](https://www.nature.com/articles/s41586-025-08744-2). Nature, 2 April 2025; DreamerV3.

A learned environment model supports policy improvement across many control tasks. The system uses trained world-model, actor, and critic networks; it is not equivalent to asking a hosted language model to imagine a future.

**Design implication:** start with explicit forecasts and observed outcomes. A trained world model is a later research branch requiring its own training environment and compute, not a prerequisite for the first Nūr experiment.

## Promising proposals that are not established solutions

### P1 Artificial Agency Program

[Csaky, Artificial Agency Program: Curiosity, Compression, and Communication in Agents](https://arxiv.org/abs/2602.24100v1). Research program and position paper, 27 February 2026.

The proposal combines bounded resources, prediction/compression, curiosity, and communication in a staged agency research program. Its broad claims about the sufficiency of a core motivation and the requirements for agency remain hypotheses.

**Design implication:** test learning-progress-based exploration against simple alternatives. Do not install one curiosity objective and present the proposed larger theory as experimentally established.

### P2 Complementary intrinsic objectives and failure modes

[Belikov, Intrinsic Motivation in Reinforcement Learning: A Research Agenda for Adaptive Self-Organisation](https://arxiv.org/abs/2609.17325v1). Perspective and experimental agenda, 15 September 2026.

The paper discusses curiosity, empowerment, learning progress, and skill discovery, emphasizing failure modes and proposing resource-constrained, multiscale experiments. Those proposed experiments are not completed demonstrations of open-ended autonomy.

**Design implication:** include unlearnable noise, already-mastered tasks, and repetitive attractors in testing. Do not assume maximizing novelty or a single intrinsic score produces sustained development.

### P3 Active inference does not eliminate initial assumptions

[Wen, The Missing Reward: Active Inference in the Era of Experience](https://arxiv.org/abs/2508.05619v1). Position paper, 7 August 2025.

The author proposes combining active inference and language models to reduce continual external reward engineering. The approach still includes a generative model and preferences; it does not demonstrate an assumption-free or value-free agent.

**Design implication:** keep active inference as a competing explanatory implementation if simpler methods fail. Changing the mathematical objective is not itself evidence of humanlike independence.

### P4 Continuous experience rather than isolated responses

[Silver and Sutton, Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf). 2025 chapter preprint and research vision.

The proposal emphasizes continuing interaction, grounded observations and consequences, and learning from experience. It is a direction for capability development, not an empirical demonstration of humanlike emotional life or freely emerging values.

**Design implication:** make environment feedback and retained learning central, while declaring how outcomes are valued. More time online without informative interaction is not the proposed learning mechanism.

## Evidence limits and research choices

The recent empirical studies offer sharper tests, not a consensus psychological architecture. Their settings range from human laboratory tasks to prompted fictional narratives, game control, and model-generated survey responses. Results cannot be transferred between those settings without testing.

For Nūr, the recommended near-term synthesis is deliberately modest: **persistent commitments, evidence-backed predictions, outcome-tested procedures, and a separately testable appraisal layer**. Human goal-selection tasks provide the first behavioral reference. More elaborate intrinsic motivation, active inference, and weight training remain alternatives evaluated only when a specific failure calls for them.

Claims about human resemblance require a declared population, task, interface, and time scale. Synthetic days are not human years. Repeated calls to one model are not a representative population of people. The plan therefore separates engineering acceptance, exploratory findings, and later confirmatory behavioral claims.
