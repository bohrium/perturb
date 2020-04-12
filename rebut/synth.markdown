We thank the reviewers R1, R2, and R4 for their constructive feedback on our
ideas and writing.  We have revised our manuscript as described below. 

**NARRATIVE AND ACCESSIBILITY**

We thank R2 for advocating for accessibility.  We now compartmentalize our core
learning-dynamical narrative from the physical ideas that inspired it.  Thus,
we plainly state our main results in a new **stand-alone** sub-section (2.2),
and we explicitly discuss the meanings of and possible pathologies in the
limits we take --- as applied to real, non-asymptotic scenarios.

We have also completely revised our tutorial appendix.
 
We also ground our future directions (lagrangian; gravity; Sect 5) in their
computational implications, integrating them with the relevant pre-conclusion
sections (3.3, 3.1).

**REAL-WORLD APPLICABILITY**

We thank R4, R1 for probing the limitations of our analysis.  Sect 4.1 now
highlights that our un-re-summed test loss predictions fail for small (eta T),
e.g. after CIFAR accuracy has improved only ~**XXX**% to convergence.  

Our main technical contribution, the re-summed force laws, agree with
experiment for **long times**.  E.g., Fig 6(left) validated our entropic force
prediction on a synthetic dataset for T=10000, during which the weights drifted
~5 times farther than the problem's natural length scale.  We now include an
analogous plot for CIFAR convnets trained for 1000 updates on 100 points;
our 3rd order predictions explain ~**XXX**% of the variance in final weights.

Though our test loss predictions eventually fail due to accumulating error, we
have shown that our force laws remain valid in real training scenarios.  Our
results --- e.g. that SGD avoids high-covariance weights and that it dis-aligns
the hessian from the current covariance --- thus equip practitioners to reason
about which minima SGD prefers.

While Newton's Laws are most falsifiable in a vaccuum, they shed qualitative
light even on automobiles with unknown friction laws.  Our theory, though most
stringently testable in simple cases, likewise illuminates real-world SGD.

**ENGLISH AND NOTATION**

We have combed through the text to reduce jargon, to prefer idioms, and to
coordinate our notation between pages, e.g.: 

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

Re "renormalization": we thank R1 for their lucid explanation, suggestion, and
questions.  Yes, "re-summation" is more apt: we don't introduce counterterms,
and we have no divergence at small scales.  We now write "re-summation".

**TODO**
is this a resummation of the diagrams as eta T-> finite, eta = small ? Could
the authors bring some intuition of why terms with only two edges are dominant?

Re "Feynman": Both our and Feynman's formalisms arise from repeated
differentiation of a sum-over-paths of a product-over-time of an exponential.
Our diagrams' fuzzy ties give rise to loops.  Like Penrose's "String Diagrams",
ours use edges to depict contraction --- we will adopt this name if so advised.

Re "valley": in the over-parameterized setting of deep networks without explicit
regularization, the training landscape's minima (by Implicit Function Theorem)
typically form a submanifold of dimension (#weights)-(#trainset).  The
literature (e.g. "Asymmetric Valleys: Beyond Sharp and Flat Local Minima" (He,
Huang, Yuan: NeurIPS 2019)) refers to these submanifolds as "valleys".

**OPAQUE CLAIMS**

As R4 highlighted, our inlined claims require discussion.  We now indicate
which claims are routine to check and which claims, justified in the
appendices, are to be taken as blackboxes.  We also more precisely phrase
claims such as those below:

Re "smoothed landscape": the diagram [[image A]] contains a subdiagram [[image
B]] = C*H; it is routine to check that [[B]] gives the leading-order loss
increase when we convolve the landscape with a C-shaped Gaussian.  Since [[A]]
connects [[B]] to the test measurement using 1 edge, it couples [[B]] to the
linear part of the test landscape --- and hence represents a displacement of
weights (away from areas where the smoothed landscape is high).  In short,
[[A]] depicts descent on a smoothed landscape.

Re "implicit metric": to define SGD or to measure a hessian's width, we need a
Riemannian metric.  Prior work on generalization and hessian-width chose this
metric arbitrarily.  We proved that when we measure hessian-width using the
metric implicit in SGD, we get a generalization gap estimate.  We thus show
that our choice of optimizer modulates the relationship between curvature and
generalization.

**CHOICE OF VENUE** 

We are inspired by prior work (e.g. "Asymptotics of Wide Networks from Feynman
Diagrams" (Dyer, Gur-Ari: ICLR 2019) and "Wasserstein Generative Adversarial
Networks" (Arjovsky, Chintala, Bottou: ICML 2017)) that communicated
substantive new mathematical tools within a conference paper's bounds.  Our
paper, now restructured thanks to reviewer feedback, may be absorbed either as
a short report of results, or, through the revised appendices, as an in-depth
discussion.
