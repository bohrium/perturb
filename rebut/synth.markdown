We thank the reviewers R1, R2, and R4 for their constructive feedback on our
ideas and writing.  

**ACCESSIBILITY**

Reviewers appreciated the novelty of the method and expressed interest in our
results but had concerns over the clarity of the exposition and the
accessibility to an ICML audience.  In response to these concerns, we are
significantly revising the presentation to highlight our key contributions and
develop our results more clearly.

* Our introduction now more clearly motivatres our approach.  We explain at the
  outset that our goal is to provide practitioners and theorists with new
  intuition about the behavior of SGD -- for example, reasoning about systematic
  biases in SGD that cause it to prefer some local minima over others.  
* In a new stand-alone section, we state our key results and discuss how they
  map onto real, non-asymptotic scenarios.

As R4 highlighted, our inlined claims require discussion.  We now indicate
which claims are routine to check and which claims are justified in the
appendices, and we clarify, e.g.:

* "smoothed landscape": the diagram [[image A]] contains a subdiagram [[image
  B]] = CH; by a routine check, [[B]] gives the leading-order loss increase
  when we convolve the landscape with a C-shaped Gaussian.  Since [[A]]
  connects [[B]] to the test measurement using 1 edge, it couples [[B]] to the
  linear part of the test landscape --- and hence represents a displacement of
  weights away from high [[B]].  In short, [[A]] depicts descent on a smoothed
  land.

* "implicit metric": to define SGD or to measure H's width, we need a metric.
  Prior work relating gengap to H-width chose a metric arbitrarily.  We proved
  that {measuring H-width with the metric used to define SGD} yields a gengap
  estimator.  Thus, the choice of optimizer modulates how curvature and gengap
  relate.

**PRACTICAL APPLICATIONS**

We thank R4, R1 for probing our work's limitations.  Since our predictions
depend only on loss data near initialization, they break down after the weight
moves far from initialization.  Our theory thus best applies to small-movement
contexts, whether for long times near an isolated minimum or for short times in
general.

Even short time (e.g. T=10) predictions are interesting because they show how
curvature and noise (and not just gradients) repel or attract SGD's current
weight.  E.g., we proved that SGD moves to dis-align H from the current C, and
that this effect dominates when the gradient is 0.  Intuitively, initial data
rarely suffices to predict long-time behavior, because landscapes can grow
arbitrarily complex.  Instead, our contribution is to quantify
counter-intuitive deterministic "forces" governing SGD's short-time dynamics.
Since our analysis works for any initialization, one may imagine SGD's
trajectory as an integral curve of the vector field given by our theory.
Insofar as practioners rely on intuitions such as that "SGD descends on the
average train loss", our force laws serve to enhance such intuition.

We re-organized our theoretical and experimental discussion to emphasize the
above.  Sect 4.1 now shows that our un-resummed test loss predictions fail for
large (eta T), e.g. after CIFAR accuracy has improved only ~1 point.
Supplementing Fig 6a (T=10000), which validated on a synthetic problem the
entropic force on a length scale ~5 times the natural length, we now
demonstrate an entropic force for CIFAR (T=1000), initialized in a valley of
minima; our eta^3 predictions explain ~**XXX**% of the variance in final
weights.

**ENGLISH AND NOTATION**

We have improved the text by reducing jargon, preferring idioms, and
consolidating our definitions, e.g.:

* We have omitted vague words such as "precision".

* On "mean-l distribution": by this we mean, "a distribution on functions with
  a well-defined expectation l."  We thank R4 for pointing out this was
  unclear.

* Sect 2.3 now defines and discusses the inter-related SGD hyperparameters
  T,E,N,B, and M.  T counts SGD updates in total.  Over E epochs through N pts
  with B pts per batch, there are (N/B) updates per epoch and thus T=E(N/B)
  updates.  E.g., for CIFAR with maximal B over E=50: T=50.
  We also defined M=E/B.  On a noiseless, linear landscape, loss depends only
  of eta, N, and M.  So it is natural to compare SGD variants of equal M. 

**TECHNICAL TERMS**

* "renormalization": we thank R1 for incisive explanation and questions.  Since
  we don't introduce counterterms and we suffer no divergence at small scales,
  we now write "resummation". :

    * "eta T to finite, eta = small": Yes!  As eta\to 0 with eta T fixed, SGD
      becomes ODE; per Rmk 1, diagrams with f fuzzy ties give the f-th order
      corrections due to non-0 eta (and hence to noise).  Still, even for a
      quadratic landscape, those un-resummed corrections involve unboundedly
      many diagrams; re-summation solves this problem.

    * "two edges are dominant": Resummed terms with d edges are order eta^d
      (despite edges now representing variable-length chains of un-resummed
      edges).  So for small eta, terms with few edges dominate.  E.g., Cor.s
      (6,7) involve (3,2)-edged diagrams.  

* "Feynman": Both our and Feynman's formalisms arise from differentiating of
  a sum-over-paths of a product-over-time of an exp.  Our diagrams' fuzzy ties
  yield loops.  Like Penrose's "String Diagrams", ours use edges to depict
  contraction --- we will adopt this name if advised.

* "valley": in the over-parameterized setting of deep networks without
  explicit regularization, the train minima (by Impl.Func.Thm.) typically form
  a submanifold of dimension (#weights)-(#trainset).  The literature (e.g. (He,
  Huang, Yuan: NeurIPS 2019)) refers to these submanifolds as "valleys".

**CHOICE OF VENUE** 

We are inspired by prior work (e.g. (Dyer, Gur-Ari: ICLR 2019) and (Arjovsky,
Chintala, Bottou: ICML 2017)) that communicated substantive new mathematical
tools within a conference paper's bounds.  We have focused our paper into a
practical report on force laws, with the surrounding theory restructured into
appendices. 
