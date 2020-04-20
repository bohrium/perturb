We thank the reviewers R1, R2, and R4 for their constructive feedback on our
ideas and writing.  

**ACCESSIBILITY**

Reviewers expressed interest in our results but had concerns over our
exposition's accessibility and clarity.  In response to these concerns, we
significantly revised our presentation:

* We explain at the outset that our goal is to provide new intuition about SGD
  dynamics, e.g. how overfitting scales w/ curvature and noise.

* We state key results in a new *stand-alone* section unobscured by physics.
  We discuss how they map onto real, non-asymptotic scenarios.

As R4 noted, our inlined claims require discussion.  We expanded on our claims
and indicate which are routine to check (rather than justified in appendices),
e.g.:

* "smoothed landscape": the diagram [[image A]] contains a subdiagram [[B]] =
  CH; by a routine check, [[B]] is the leading-order loss increase when we
  convolve the landscape w/ a C-shaped Gaussian.  Since [[A]] connects [[B]]
  to the test measurement using 1 edge, it couples [[B]] to the linear part of
  the test landscape and hence represents a displacement of weights away
  from high [[B]].  In short, [[A]] depicts descent on a smoothed land.

* "implicit metric": to define SGD or to measure H's width, we need a metric.
  Prior work on gengap and H-width chose metrics arbitrarily.  We show
  that {measuring H-width w/ the metric used to define SGD} yields a gengap
  estimator.  So optimization modulates how curvature and gengap relate.

**PRACTICAL APPLICATIONS**

As our predictions depend only on loss data at init., our theory best applies
to small net weight displacements, e.g. for long times near an isolated minimum
or for short times in general.

Short-time theory reveals how curvature and noise --- and not just averaged
gradients --- affect learning.  E.g: regions flat (w.r.t. the current C)
attract SGD.  Meteorologists understand how storms arise despite long-term
forecasting's intractability; our theory quantifies the counter-intuitive
dynamics governing SGD's short-time behavior.  In accounting for noise, our
work enhances noiseless intuitions relied on by practitioners --- e.g. "SGD
descends on the train loss" --- by giving new force laws valid in each
short-term patch of SGD's trajectory.

We show that our un-resummed test loss predictions fail beyond small eta·T
(e.g. after CIFAR acc. moves ~1 point).  Next to Fig 6a (T=10000 entropic
force on synthetic data), we now demonstrate an entropic force for CIFAR
(T=1000), init'd in a valley; our eta^3 predictions explain ~**XXX**% of the
variance in final weights.

Fine-tuners such as MAML seek models near local minima tunable to new
data within small T, a setting matched to the assumptions of our theory.

**ENGLISH AND NOTATION**

We have cut jargon, consolidated our definitions, and:

* omitted vague words such as "precision".

* decompressed "mean-l distribution" to "a distribution (on functions)
  with expectation l." 

* in a single subsect, defined and relatd the SGD parameters T,E,N,B,M:
  T counts updates in total.  M=E/B.  Over E epochs through N pts with B
  pts per batch, SGD updates (N/B) times epoch for a total of T=E(N/B).
  For CIFAR over E=50 with maximal B: T=50.

**TECHNICAL TERMS**

* "renormalization": we thank R1 for their incisive comments.  Since we suffer
  no divergence at small scales, we now write "resummation".

    * "eta·T finite, eta small": Yes!  As eta→0 with eta·T fixed, SGD
      becomes ODE, and summing diagrams with f fuzzy ties give the order-f
     corrections due to non-0 eta.  Still, even for a quadratic landscape, each
     such correction involves unboundedly many diagrams; re-summation remedies
     this.

    * "two-edged terms dominate": for fixed eta·T, each re-summed term with d
      edges is order eta^d.  So for small eta, terms with few edges dominate
     (e.g., Cor.s (6,7) involve (3,2)-edged diagrams).  But for fixed T,
     re-summed terms are often order eta^e with e<d.  Both limits are
     interesting and we clarify our discussion accordingly. 

* "Feynman": our and Feynman's diagrams arise from differentiating a
  sum-over-paths of a product-over-time of an exp.  Our diagrams' fuzzy ties
  yield loops.  As in Penrose's "String Diagrams", edges depict contraction
  --- we will adopt this name if advised.

* "valley": in the over-parameterized setting of deep networks w/o 
  explicit regularization, the train minima (by Impl.Func.Thm.) typically form
  a submanifold of dimension (#weights)-(#trainset).  The literature (e.g. (He,
  Huang, Yuan: NeurIPS 2019)) refers to these submanifolds as "valleys".

**CHOICE OF VENUE** 

We are inspired by prior work (e.g. (Dyer, Gur-Ari: ICLR 2019) and (Arjovsky,
Chintala, Bottou: ICML 2017)) that communicated substantive new mathematical
tools within a conference paper's bounds.  We have focused our paper into a
practical report on force laws, and we will restructure the surrounding theory
into the appendices. 
