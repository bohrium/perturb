We thank the reviewers for grappling with our terminology and for their
constructive feedback on our ideas and writing.  We have revised our manuscript
both globally and locally as described below. 

**NARRATIVE AND ACCESSIBILITY**

        FILL IN

    ACCESSIBILITY

**REAL-WORLD APPLICABILITY**

Our revised text now highlights our un-renormalized (un-re-summed) estimates'
durations of applicability: on image sets, they fail after accuracy has
improved only ~ **XXX** % of the way from initial to converged value.

Our main technical contribution, the re-summed theory, yields non-divergent
**long-time** predictions.  For example, we correctly predicted the entropic
force on a synthetic dataset for thousands of SGD updates, during which the
weights drifted an order of magnitude farther than the problem's natural length
scale. 

The theory shines qualitative insight on large networks trained to convergence,
e.g. that SGD tends to avoid high-covariance weights, that it tends to
dis-align the hessian from the current covariance, and that these effects drive
SGD to prefer some minima to others.  Our experiments merely complement our
mathematical proofs (e.g., to ensure we didn't overlook any factors of 2),
hence raising our confidence in these insights.

While Newton's Laws are most stringently testable in a vaccuum, they shape our
understanding even of automobiles with unknown friction laws.  Our theory,
though most stringently testable in simple cases, likewise illuminates deep
learning.

**ENGLISH AND NOTATION**

We have combed through the text to reduce jargon, to prefer idioms, and to
coordinate our notation between pages, e.g.: 

We have omitted vague words such as "precision" in "precision test"..

By a "mean-l distribution", we meant a "distribution over function space that
has a well-defined expectation, for convenience called l".  Our revision spells
this out explicitly.  Likewise for our overall framework: e.g., our new use of
subscripts distinguishes between the (potentially infinite) population of
datapoints and the (finite) set of train samples. 

We used inter-related SGD hyperparameters T, E, N, B, and M.  The new Section
1.3 now defines and conceptually compares them.  T counts SGD updates in
total.  Over E epochs through N points with B points per batch, there are (N/B)
updates per epoch and thus T=E*(N/B) updates.  E.g., for CIFAR with maximal
batch size over 50 epochs, T would be 50.  We also defined M=E/B.  On a
noiseless, linear landscape, batch-size B SGD run for E epochs matches
batch-size 1 SGD run for M epochs.  So it is natural to compare SGD variants
that have the same M. 

**TECHNICAL TERMS**

Yes, "re-summation" is more apt: we don't introduce counterterms.  Our revised
text uses "re-summation".

Why "Feynman"?  Both our and Feynman's formalisms arise from repeated
differentiation of a sum-over-paths of a product-over-time of an exponential.
Our diagrams typically have nontrivial topology due to the fuzzy ties.  Like
Penrose's "String Diagrams", ours use edges to depict contraction --- we will
adopt this name if so advised.

"Valley": in the over-parameterized setting of deep networks without explicit
regularization, the training landscape's minima (by Implicit Function Theorem)
typically form a "shape" of dimension p-n, where p counts weights and n counts
train points.  Much prior work (e.g. "Asymmetric Valleys: Beyond Sharp and Flat
Local Minima" (He, Huang, Yuan: NeurIPS 2019)) refers to these shapes as
"valleys".

**OPAQUE CLAIMS**

As Reviewer 4 highlighted, we presented many of our claims with confusingly
sparse discussion.  Our revised text now indicates which claims are routine to
check and which claims, justified in the appendices, are to be taken as
blackboxes.  We also more precisely phrase claims such as those below:

Re: "descent on a smoothed landscape": the diagram [[image]] contains a
subdiagram [[image]] = covariance * hessian; it is routine to check that
[[subdiagram image]] measures the leading-order increase in loss when we blur
the loss landscape with a  Gaussian filter shaped like the covariance.  To get
the whole diagram [[image]], we connect that subdiagram [[image]] to the test
loss measurement using 1 edge.  The term represents a coupling (from the
subdiagram) to the linear part of the test landscape --- and hence represents a
displacement of the weights.  We conclude that this diagram [[image]] depicts a
displacement away from areas where the smoothed landscape is high.  

Re: the "inherent metric": to define SGD or to measure a hessian's width,
we need a Riemannian metric.  Prior work on generalization and hessian-width
chose this metric arbitrarily we proved that when we measure a hessian's width
using the metric implicit in SGD, we get a generalization gap estimate.  We
thus show that our choice of optimizer modulates the relationship between
curvature and generalization.

**CHOICE OF VENUE** 

We thank Reviewer 4 for raising the question of venue.  We are inspired by
prior work (e.g. "Asymptotics of Wide Networks from Feynman Diagrams" (Dyer,
Gur-Ari: ICLR 2019) and "Wasserstein Generative Adversarial Networks"
(Arjovsky, Chintala, Bottou: ICML 2017)) that communicated new mathematical
tools within a conference paper's bounds.  Our paper, now restructured thanks
to reviewer feedback, may be absorbed either as a short report of results, or,
through the revised appendices, as an in-depth discussion.
