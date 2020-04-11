# Rebuttals for SGD Paper

## SYNOPSES

**R1** The paper studies the statistics of quantities such as the loss function
at time t as a function of hyperparameters and measurements at initialization.

**R2** A diagram based approach for understanding SGD, which then illustrates
the generalization properties of the algorithm.

**R4** The paper models the test loss and generalisation gap of SGD for small
learning rates and small numbers of epochs.  It uses perturbation calculus and
diagram formalism (making an analogy to Feynman diagrams in physics) to
enumerate and manipulate relevant terms. The resulting estimators are tested on
image datasets and synthetic loss landscapes.

## MERITS

**R1** This paper presents a novel way of analyzing the SGD dynamics which can
be in principle be very powerful to understand better properties of trained
neural networks such as generalization or optimal hyperparameters.

**R2** The paper proposes an interesting way of analyzing SGD based on
diagrams.  This diagram-based analysis seems very novel and might invoke
further research.  The conclusions about wide/sharp minima are interesting. 

**R4**
1. New insights into the behaviour of SGD at early training stage.
2. Popularisation of diagram formalisms in machine learning.

## EVALUATIONS

**R1** Very good paper, I would like to see it accepted; I tried to check the
important points carefully. It is unlikely, though possible, that I missed
something that could affect my ratings.  I really like the paper and I would
have marked it as outstanding if the presentation was clearer: where are the
authors going, what are the limitations and so on. In its current form the
paper is hard to read. 

        Thanks.  I will work on the presentation, both globally (order of
        exposition, pedagogical examples, motivation, future directions,
        limitations) and locally (improving grammar and reducing jargon).

**R2** Borderline paper, but the flaws may outweigh the merits; not my area, or
the paper was hard for me to understand.  

**R4** Below the acceptance threshold, I would rather not see it at the
conference; not my area, or the paper was hard for me to understand.

## DETAILED FEEDBACK

**R1** The paper presents a rather novel approach to SGD dynamics presenting a
set of diagrammatic tools to compute the loss function at any order in the
learning rate.

Several applications are mentioned in the paper but I find lacking clearer
experiments illustrating the power of the mechanism. For example, in the
current presentation, it is not clear what is the time scale at which this
formalism hold, for example in figure5 left, how does this time compare with
the required training time, what accuracy is obtained in this time...? 

        TODO

I find the notation renormalization a little confusing, and I think what the
authors are doing is rather resummation, this is something which is often done
in physics where we have a series expansion and we take a limit of such a
series so that only some diagrams matter
(in physics language: renormalization
comes from a lack of understanding in our current system it requires a "UV"
divergence and adding counterterms/regularization, while IR divergences come
from the breakdown of perturbation theory and are treated by resumming
diagrams, the "thermal mass" is an example of such resummation).
If this is
what is happening it might be more helpful to put it in this way: is this a
resummation of the diagrams as eta T-> finite, eta = small ? Could the authors
bring some intuition of why terms with only two edges are dominant? 

        Thanks for the lucid explanation!  What we called "renormalization"
        sounds indeed like resummation: rather than introducing counterterms, we
        evaluated the same sum by analytical trickery.  In an (optional
        and convenient) last step, we approximated sums over integer-length
        chains by integrals over continuous-length chains, hence obtaining a
        scale-invariant expression appropriate for large scales (times).
        However, the theory experiences no catastrophe at very small scales, 
        so IR is indeed a more appropriate analogue.
        We will revise the text to use the word "resummation" throughout.

        [Our backgrounds are in pure math, so our attempt to concretize the
        concepts using physical imagery has apparently backfired!]

**R2** The paper is based on non-standard tools in machine learning. Even
though a tutorial-like information is given in the supplementary document, I am
afraid that the paper will not be accessible to the general ML audience. That
is why I am suggesting a weak rejection.

I must admit that I am not able to appreciate the paper in details, hence my
confidence.

The paper seems to be well-written, at least grammatically. However, as I am
not familiar with the tools used in the paper, I am not able to judge the
novelty of the approach. My current opinion is that the paper has interesting
ideas, but would not be accessible for the ICML audience. If the other
reviewers have different opinions I would be willing to update my score. 

**R4** I really struggled to follow the Author's exposition. They crammed a lot
of technical arguments into short paragraphs. They often use terms before
defining them, and have a tendency to use jargon. The Supplementary Material,
despite being 18 pages long, did not help me much. It is also a bit chaotic:
the proof of Corollary 1 is split between Appendices A and B.2.

I am not sure about how relevant the results of this analysis are for practical
applications. In the test on Fashion-MNIST (Figure 5, left), the Authors ran
SGD for approx. 10 datapoints (as explained in Appendix G.3). This does not
seem to me to be a very realistic scenario.

        Good point.  We will list and emphasize the regimes for the breakdown
        of our estimates.  Our un-renormalized predictions typically diverge
        from the truth after small eta*T, so we test them on impractically few datapoints.
        One of our main contributions is the renormalized theory, which in a
        positive-hessian setting yields non-divergent long-time predictions.  For
        example, we performed our nonconservative entropic force experiments on a
        synthetic dataset for thousands of gradient updates.  A limitation of the
        renormalized theory is that it requires a positive and exactly known hessian
        (at initialization) in order to make quantitative rather than qualitative
        predictions.  We view these small-scale experiments as complementing
        our mathematical proofs (e.g., to ensure we overlooked no factors of 2)
        in order to raise our confidence in our theory's qualitative
        insights about real systems such as image classifiers trained to
        convergence.  An example of such an insight is that SGD tends to
        move away from high-covariance locations and also to dis-align the
        hessian from the current covariance, and that these effects explain why
        SGD prefers some minima to others.

The manuscript contains occasional grammatical errors, and its style should be
improved (see examples) below.

        Thank you for the detailed list of example errors.  We will comb
        through the paper to streamline grammar and minimize jargon. 

1. Are you confident that a conference paper with 9-page limit is an
   appropriate format for the publishing your results? I think your ideas and
work deserve much more space to be clearly explained to the reader who is not
an expert in perturbation calculus and diagram methods.

        Thanks for the thoughtful question.  We are not confident that a short
        conference paper is the most appropriate format for this content.  That
        said, we are inspired by prior work, for instance
            "Asymptotics of Wide Networks from Feynman Diagrams" (Dyer, Gur-Ari 2019),
        that communicated sophisticated new mathematical tools within the bounds
        of a conference paper by crafting a clean interface between body and
        appendices.

2. It would be much easier to understand your method if, before describing the
   advanced formulation (which perhaps could be published separately in a
journal?), you provided a very simple, step-by-step pedagogical example of the
its application.

3. (a) Could you comment on the applicability of your results to practical use
   cases, e.g. to image classifiers trained in realistic settings?

3. (b) I think it would be worth investing some time in improving the wording of
   the paper.

(a) "How does SGD select from among a valley of
minima" - why a valley? did you mean "select from multiple local minima"?

        Yes, the general situation is an unstructured set of local minima.  In
        the over-parameterized setting of deep networks without explicit
        regularization, the training landscape's minima (by Implicit Function Theorem)
        typically form a "shape" of dimension p-n, where p counts weights and n counts
        train points.  The literature colloquially refers to these submanifolds as
        "valleys".

        The test landscape typically has fewer such degeneracies.  We will
        clarify our discussion about the train landscape vs test landscape and our
        predictions about their relationship.

(b) "We verify our predictions via precision order-η 3 tests" - why "precision"?
what do you mean here? 

        We will omit "precision" and other vague words.  We meant to indicate that our asymptotics
        did not neglect constant factors (as big O notation does).
        We will emphasize this near the text's beginning and leave the question
        of whether this signals "precision" to the reader's judgement.

(d) "mean- l distribution over smooth functions l n" (Sec. 2.1) -
what does "mean-l" mean?

        We meant that the mean of the distribution over functions l_n is equal
        to another function called l.  We will spell this out.  We will also revise
        the subscripts on the l's to better distinguish the population of
        all datapoints vs a finite train sample.  Here, we meant that the mean
        loss that we wish to predict is taken over the whole (test) population
        but that we train on only a finite sample of such datapoints.

(c) "Formalism" should not be capitalised in the title
of Sec. 1.1

(e) "For instance, simplification 3 is licensed
because" (Definition 1) - I think it should be "is allowed"

f) "root r
participates in no fuzzy edge" (Theorem 1) - I think "root r does not
participate in a fuzzy edge" would be more idiomatic English

        Thank you.  These specific hints help us search for errors in general.

4. Is it justified to call your diagrams "Feynman diagrams"? The latter have a
   more complicated structure (e.g. loops) than yours.

        We chose the name ``Feynman Diagram'' because the two diagram
        formalisms arise in mathematically analogous ways, namely from repeated
        differentiation of a sum-over-paths of an exponential of a sum-over-time.  Due
        to the fuzzy ties, the diagrams can have nontrivial topology.  We also cited
        Penrose's "String Diagrams", which are a general class of notations united by
        their depictions of tensor contraction.  If recommended, we will adopt this
        name.  

5. What is the parameter T? How is it related to the number of epochs and batch
   size? If I train a CIFAR-10 model for 50 epochs, what is my T?

        We will clearly define the inter-related hyperparameters T, E, N, B,
        and M near the beginning of our revision.

        T is the total number of gradient updates.  If there are N points, B
        points per batch, and E epochs, then there will be (N/B) updates per
        epoch and therefore T=E*(N/B) steps total; for the given CIFAR example
        with full batch size, T would be 50.
        These hyperparameters also relate to what we called M=E/B, which
        is the number of epochs one would have to run 1-batch SGD in order
        to get equivalent dynamics --- assuming a noiseless and linear loss
        landscape.  We chose our set of hyperparameters (N, B, M) in order to
        make such comparisons easy to describe. 

6. The paper contains many opaque and briefly explained claims. E.g. in Section
   4.4 you write "Intuitively, the presence of the term [diagram] in our test
loss expansion indicates that SGD descends on a covariance- smoothed
landscape." Maybe the problem is on my side, but this is not intuitive to me at
all. Same for "The resulting Stabilized TIC (STIC) uses the metric implicit in
gradient descent to threshold flat from sharp minima" - what does it mean?

        We will work to set our claims more coherently into the logic of the
        paper.  For the two mentioned instances:   

        (a) Here we were unclear indeed!  That diagram contains a subdiagram
            that we may read as a covariance contracted with a hessian, i.e.
            the net increase in loss upon blurring with respect to the 
            covariance.  The diagram is that subdiagram together with one edge
            that connects to the test loss measurement (i.e. couples to the
            linear part of the test landscape, i.e. represents a displacement
            of the weights).  So we read this diagram as depicting a
            displacement away from areas where the smoothed landscape is high. 

        (b) To define gradient descent, we require a Riemannian metric.  Work
            on descent often notationally suppresses this metric.  A scalar notion of a
            hessian's sharpness also requires a Riemannian metric.  Prior empirical work
            investigates whether sharp or flat minima generalize better, but the notion of
            sharpness therein typically depends on an arbitrary choice of parameterization.  We
            show that when we use the metric implicit in gradient descent to measure
            a hessian's sharpness, we get a generalization estimate.  This emphasizes
            that the relationship of curvature and generalization is a relationship that
            depends on choice of optimizer.
