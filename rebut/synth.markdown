**PRESENTATION AND NARRATIVE**

We thank the reviewers for their constructive feedback on the prose; we will
revise the manuscript both globally and locally.  Specifically, we will

--- by the presentation 
    --- order of exposition
    --- pedagogical examples
    --- motivation
    --- future directions
    --- limitations
and locally
    ---
    ---
    --- reducing jargon

**ENGLISH AND NOTATION**

We thank the reviewers for detecting several patterns of error in our writing.
We will comb through the text to reduce jargon, to write idiomatically, and to
coordinate our notation between pages. 

Regarding the specific errors mentioned: we wrote "precision test" to hint 
that our theory predicts constant factors (unlike big O notation).  We will
omit vague words such as "precision".

When we considered a distribution with "mean l", we meant to fix a distribution
over function space that has a well-defined expectation, for convenience called
l.  Our revision spells this --- and the rest of our framework --- out
explicitly.  For example, our new use of subscripts distinguishes between the
(potentially infinite) population of datapoints and the (finite) set of train
samples. 

We parameterized SGD variants by inter-related numbers T, E, N, B, and M.  Our
revised Section 1.3 defines and conceptually compares these hyperparameters.  T
counts gradient updates in total.  Over E epochs through N points with B points
per batch, there are (N/B) updates per epoch and thus T=E*(N/B) updates.  For
CIFAR with maximal batch size over 50 epochs (Reviewer 4's example), T would be
50.  We also use M=E/B.  On the simplest possible landscape --- where loss is
noiseless and linear in weights --- batch-size B SGD run over E epochs matches
batch-size 1 SGD run for M epochs.  Hence, it is natural to compare SGD
variants that have the same M.  

We thank Reviewer 4 for highlighting our unidiomatic prose.  We will simplify
and standardize our writing.  

**TECHNICAL TERMS**

We thank the reviewers for thinking through our terminology. 

"Resummation" is indeed more apt: we don't introduce counterterms.  In an
(optional and convenient) last step, we approximate sums over integer-length
chains by integrals over continuous-length chains, hence obtaining a
scale-invariant expression appropriate for large scales (times).  However, the
theory experiences no catastrophe at very small scales, so IR is indeed a
fitter analogue.  Our revised text uses "resummation".

We chose the name ``Feynman Diagram'' because the two diagram formalisms arise
in mathematically analogous ways, namely from repeated differentiation of a
sum-over-paths of a product-over-times of an exponential.  Due to the fuzzy
ties, the diagrams typically have nontrivial topology.  We also cited Penrose's
"String Diagrams", which are a more general class of graphics united by their
depiction of tensor contraction.  If recommended, we will adopt this name.  

"Valley": In the over-parameterized setting of deep networks without explicit
regularization, the training landscape's minima (by Implicit Function Theorem)
typically form a "shape" of dimension p-n, where p counts weights and n counts
train points.  The literature colloquially refers to these submanifolds as
"valleys".  (The test landscape typically has fewer such degeneracies.  Our
revised text clarifies the distinction between train and test landscapes
as well as our predictions about their relationship.)

**OPAQUE CLAIMS**

We thank Reviewer 4 for illustrating how our claims are confusing to follow.
Our revision helps the reader by indicating which claims we consider as routine
to check and which claims are justified in the appendices and for the moment
to be taken as blackboxes.  We also phrase claims such  that about "descent on
a smoothed landscape" more precisely as follows.

Regarding "descent on a smoothed landscape":
The diagram [[image]] contains a subdiagram [[image]] that represents a
covariance twice contracted with a hessian; it is routine to check that
[[subdiagram image]] measures the leading-order increase in loss when we blur
the loss landscape with a  Gaussian filter shaped like the covariance.  To get
the whole diagram [[image]], we connect that subdiagram [[image]] to the test loss
measurement using 1 edge.  The term represents a coupling (from the subdiagram)
to the linear part of the test landscape --- and hence represents a
displacement of the weights.  We conclude that this diagram [[image]] depicts a
displacement away from areas where the smoothed landscape is high.  

Regarding the "inherent metric":
To define gradient descent requires a Riemannian metric.  Work on descent often
notationally suppresses this metric.  Scalar notions of a hessian's width
also require a metric.  Prior empirical work investigates whether wide or
sharp minima generalize better, but the notion of sharpness therein typically
depends on an arbitrary choice of parameterization.  We proved that when we use
the metric implicit in gradient descent to measure a hessian's width, we get a
generalization gap estimate.  This emphasizes that the relationship between
curvature and generalization is a relationship that depends on our choice of
optimizer.

**REGIMES OF APPLICABILITY**

Good point.  We will list and emphasize the regimes for the breakdown of our
estimates.  Our un-renormalized predictions typically diverge from the truth
after small eta*T, so we test them on impractically few datapoints.  One of our
main contributions is the renormalized theory, which in a positive-hessian
setting yields non-divergent long-time predictions.  For example, we performed
our nonconservative entropic force experiments on a synthetic dataset for
thousands of gradient updates.  A limitation of the renormalized theory is that
it requires a positive and exactly known hessian (at initialization) in order
to make quantitative rather than qualitative predictions.  We view these
small-scale experiments as complementing our mathematical proofs (e.g., to
ensure we overlooked no factors of 2) in order to raise our confidence in our
theory's qualitative insights about real systems such as image classifiers
trained to convergence.  An example of such an insight is that SGD tends to
move away from high-covariance locations and also to dis-align the hessian from
the current covariance, and that these effects explain why SGD prefers some
minima to others.

**CHOICE OF VENUE** 

We thank Reviewer 4 for the thoughtful question.  We are not confident that a
short conference paper is the most appropriate format for this content.  That
said, we are inspired by prior work, for instance "Asymptotics of Wide Networks
from Feynman Diagrams" (Dyer, Gur-Ari 2019) or "Wasserstein Generative
Adversarial Networks" (Arjovsky, Chintala, Bottou 2017), that communicated new
mathematical tools within the bounds of a conference paper.  We believe that
our paper, now revised thanks to reviewer feedback, can provide insight both as
a short report of results and, through the revised appendices, as an in-depth
discussion.
