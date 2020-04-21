We thank the reviewers R1, R2, and R4 for their constructive feedback on our
ideas and writing. 

**ACCESSIBILITY**

Reviewers expressed interest in our results but had concerns over our
exposition's accessibility and clarity. In response to these concerns, we
thoroughly revised the paper; we now:
* explain in the intro that our goal is to offer new intuition about SGD
 dynamics, e.g. how overfitting scales w/ C and H.
* state key results in a new *stand-alone* section unobscured by physics and
 discuss how they map onto real, non-asymptotic scenarios.
* cut jargon. E.g. we introduce our formalism via:
 "Consider running SGD on N train points for T steps, starting at a weight θ0.
 Our method expresses the expectation (over randomly sampled train sets) of
 quantities such as the final weight (or test or train loss) as a sum of
 diagrams, where each diagram evaluates to a statistic of the loss landscape
 at initialization."

As R4 noted, some claims required discussion. We expand on these claims, e.g.:
* "smoothed landscape": the diagram [[image A]] contains a subdiagram [[B]] =
 CH; by a routine check, [[B]] is the leading-order loss increase when we
 convolve the landscape w/ a C-shaped Gaussian. Since [[A]] connects [[B]]
 to the test measurement using 1 edge, it couples [[B]] to the linear part of
 the test landscape and hence represents a displacement of weights away
 from high [[B]]. In short, [[A]] depicts descent on a smoothed land.
* "implicit metric": to define SGD or to measure H's width, we need a metric.
 Prior work on gengap and H-width chose metrics arbitrarily. We show
 that {measuring H-width w/ the metric used to define SGD} yields a gengap
 estimator. So optimization modulates how curvature and gengap relate.

**PRACTICAL APPLICATIONS**

As our predictions depend only on loss data at init., our theory best applies
to small net weight displacements, e.g. for short times in general or long
times near an isolated minimum.

Short-time theory reveals how curvature and noise (not just averaged gradients)
affect learning. E.g: regions flat (w.r.t. the current C) attract SGD.
Meteorologists understand how storms arise despite long-term forecasting's
intractability; our theory quantifies the counter-intuitive dynamics governing
SGD's short-time behavior. Our work enhances noiseless intuitions relied on by
practitioners (e.g. "SGD descends on the train loss") by giving new force laws
valid in each short-term patch of SGD's trajectory.

We now show our un-resummed predictions failing beyond small ηT (e.g. after
CIFAR accu. moves ~1 point). Next to Fig 6a (successful T=10000 prediction on
synthetic data), we now show an entropic force for CIFAR (T=1000) init'd in a
valley. Our η^3 predictions explain ~**XXX**% of the variance in final weights.

Fine-tuners such as MAML seek models near local minima tunable to new
data within small T, a setting matched to the assumptions of our theory.

**TERMINOLOGY**

* "renorm.": we thank R1 for their incisive comments. We have no
 UV divergences, so we now use "resummation".
  * "eta·T finite, η small": Yes! As η→0 with ηT fixed, SGD becomes ODE, and
   summing diagrams with f fuzzy ties give the order-f corrections due to
   η⪈0. Still, even for a quadratic landscape, each such correction
   involves unboundedly many diagrams; resummation remedies this.
  * "2-edged terms dominate": for fixed ηT, each resummed term with d
   edges is order η^d. So for small η, terms with few edges dominate
   (e.g., Cor 6 has a 3-edged diagram). But for fixed T, resummed terms
   are often order η^e with e⪇d. Both limits have interest; we clarify our
   discussion of them. 
* "Feynman": our and Feynman's diagrams arise from differentiating a
 sum-over-paths of a product-over-time of an exp. Our diagrams' fuzzy ties
 yield loops. As in Penrose's "String Diagrams", edges depict contraction; we
 will adopt this name if advised.
* "valley": in the over-param'd setting of deep networks w/o explicit
 regularization, the train minima (by Impl.Func.Thm.) typically form a
 submanifold of dimension (#weights)-(#trainset). The literature (e.g. (He,
 Huang, Yuan: NeurIPS 2019)) refers to these submanifolds as "valleys".

we:
* omit vague words, e.g. "precision".
* decompress "mean-l distribution" to "a distribution (on functions)
 with expectation l." 
* consolidate definitions, e.g. of the SGD parameters T,E,N,B,M: T counts
 updates in total. M=E/B. Over E epochs through N pts with B pts per batch,
 SGD updates (N/B) times epoch for a total of T=E(N/B).
 For CIFAR over E=50 with maximal B: T=50.

**VENUE** 

We are inspired by prior work ((Dyer, Gur-Ari: ICLR 2019) and (Arjovsky,
Chintala, Bottou: ICML 2017)) that communicated substantive new mathematical
tools within a conference paper's bounds. We've focused our paper into a
practical report on force laws, and we will restructure the surrounding theory
into the appendices. 
