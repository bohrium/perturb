We thank the reviewers R1, R2, and R4 for their constructive feedback on our
ideas and writing.  We have revised our manuscript as described below. 

**ACCESSIBILITY**

We thank R2 for championing accessibility.  We now isolate our core SGD
narrative from its physical inspiration.  We plainly state key results in a new
stand-alone Sect (3.2) and we discuss how our limits map onto real,
non-asymptotic scenarios.  We systematize our tutorial appendix.  Instead of
the example (Sect 1.2, now an appendix), our intro now overviews our gameplan
and concrete goals: to derive force laws (e.g. that SGD avoids high-C areas) 
that may usefully aid the intuition of practitioners.
 
As R4 highlighted, our inlined claims require discussion.  We now indicate
which claims are routine to check and which claims are justified in the
appendices.  We also more precisely phrase claims such as those below:

RE: "smoothed landscape": the diagram [[image A]] contains a subdiagram [[image
B]] = C*H; by a routine check, [[B]] gives the leading-order loss increase when
we convolve the landscape with a C-shaped Gaussian.  Since [[A]] connects [[B]]
to the test measurement using 1 edge, it couples [[B]] to the linear part of
the test landscape --- and hence represents a displacement of weights away from
high [[B]].  In short, [[A]] depicts descent on a smoothed landscape.

RE: "implicit metric": to define SGD or to measure H's width, we need a metric.
Prior work relating gengap to H-width chose a metric arbitrarily.  We proved
that {measuring H-width with the metric used to define SGD} yields a gengap
estimator.  We thus show that the choice of optimizer modulates how curvature
and gengap relate.

**REAL-WORLD APPLICABILITY**

We thank R4, R1 for probing the limitations of our analysis.  Sect 4.1 now
shows that our un-resummed test loss predictions fail for small (eta T), e.g.
after CIFAR accuracy has improved only ~**XXX** points.  We expected this due
to accumulating error.

Our resummed force laws agree with experiment for large T.  E.g., Fig 6.a
showed an experiment on synthetic data with T=10000, during which the weights
drifted ~5x farther than the problem's length scale.  We now present an
analogous plot for CIFAR (T=1000, N=100); our eta^3 predictions explain
~**XXX**% of the variance in final weights.

As our force laws (e.g. that SGD avoids high-C weights and that it dis-aligns
the H from the current C) are valid in real training scenarios, they equip
practitioners to reason about the minima SGD prefers.

**ENGLISH AND NOTATION**

We have improved the text by reducing jargon, preferring idioms, and
coordinating our notation between pages, e.g.: 

We have omitted vague words such as "precision" in "precision test".

By a "mean-l distribution", we meant a "distribution over function space that
has a well-defined expectation, for convenience called l".  We now write this
explicitly.  We also use subscripts to distinguish between the test population
of datapoints and the finite set of train samples. 

Sect 2.3 now defines and conceptually compares the inter-related SGD
hyperparameters T, E, N, B, and M.  T counts SGD updates in total.  Over E
epochs through N points with B points per batch, there are (N/B) updates per
epoch and thus T=E*(N/B) updates.  E.g., for CIFAR with maximal batch size over
50 epochs, T would be 50.  We also defined M=E/B.  On a noiseless, linear
landscape, batch-size B SGD run for E epochs matches batch-size 1 SGD run for M
epochs.  So it is natural to compare SGD variants of equal M. 

**TECHNICAL TERMS**

RE: "renormalization": we thank R1 for their lucid explanation and questions.
Yes, "resummation" is more apt: we don't introduce counterterms, and we suffer
no divergence at small scales.  We now write "resummation" and clarify as
below:

As eta\to 0 with eta T fixed, SGD becomes ODE; per Rmk 1, diagrams with f fuzzy
ties give the f-th order corrections due to non-0 eta (and hence to noise).
Still, even for a quadratic landscape, those un-resummed corrections involve
unboundedly many diagrams; our re-summation solves this problem.   

Resummed terms with d edges are order eta^d (despite edges now representing
variable-length chains of un-resummed edges).  So for small eta, terms with few
edges dominate.  E.g., Cor.s (6,7) involve 3,2)-edged diagrams.  

RE: "Feynman": Both our and Feynman's formalisms arise from differentiating of
a sum-over-paths of a product-over-time of an exp.  Our diagrams' fuzzy ties
yield loops.  Like Penrose's "String Diagrams", ours use edges to depict
contraction --- we will adopt this name if advised.

RE: "valley": in the over-parameterized setting of deep networks without
explicit regularization, the train minima (by Impl.Func.Thm.) typically form a
submanifold of dimension (#weights)-(#trainset).  The literature (e.g. (He,
Huang, Yuan: NeurIPS 2019)) refers to these submanifolds as "valleys".

**CHOICE OF VENUE** 

We are inspired by prior work (e.g. (Dyer, Gur-Ari: ICLR 2019) and (Arjovsky,
Chintala, Bottou: ICML 2017)) that communicated substantive new mathematical
tools within a conference paper's bounds.  We have restructured our paper to be
absorbed either as a short report of results, or, through the revised
appendices, as an in-depth discussion.
