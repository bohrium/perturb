# Reviews of SGD Paper

## SYNOPSES

The paper studies the statistics of quantities such as the loss function at
time t as a function of hyperparameters and measurements at initialization.  A
diagram based approach for understanding SGD, which then illustrates the
generalization properties of the algorithm.  The paper models the test loss and
generalisation gap of SGD for small learning rates and small numbers of epochs.
It uses perturbation calculus and diagram formalism (making an analogy to
Feynman diagrams in physics) to enumerate and manipulate relevant terms. The
resulting estimators are tested on image datasets and synthetic loss
landscapes.

## MERITS

This paper presents a novel way of analyzing the SGD dynamics which can be
in principle be very powerful to understand better properties of trained neural
networks such as generalization or optimal hyperparameters.  The paper proposes
an interesting way of analyzing SGD based on diagrams.  This diagram-based
analysis seems very novel and might invoke further research.  The conclusions
about wide/sharp minima are interesting. 

1. New insights into the behaviour of SGD at early training stage.
2. Popularisation of diagram formalisms in machine learning.

## EVALUATIONS

### R1 

Very good paper, I would like to see it accepted; I tried to check the
important points carefully. It is unlikely, though possible, that I missed
something that could affect my ratings.  I really like the paper and I would
have marked it as outstanding if the presentation was clearer: where are the
authors going, what are the limitations and so on. In its current form the
paper is hard to read. 

### R2

Borderline paper, but the flaws may outweigh the merits; not my area, or
the paper was hard for me to understand.  I really like the paper and I would
have marked it as outstanding if the presentation was clearer: where are the
authors going, what are the limitations and so on. In its current form the
paper is hard to read. 

### R4

Below the acceptance threshold, I would rather not see it at the
conference; not my area, or the paper was hard for me to understand.  I really
like the paper and I would have marked it as outstanding if the presentation
was clearer: where are the authors going, what are the limitations and so on.
In its current form the paper is hard to read. 

## DETAILED FEEDBACK

### R1 

The paper presents a rather novel approach to SGD dynamics presenting a set
of diagrammatic tools to compute the loss function at any order in the learning
rate.  -Several applications are mentioned in the paper but I find lacking
clearer experiments illustrating the power of the mechanism. For example, in
the current presentation, it is not clear what is the time scale at which this
formalism hold, for example in figure5 left, how does this time compare with
the required training time, what accuracy is obtained in this time...?  - I
find the notation renormalization a little confusing, and I think what the
authors are doing is rather resummation, this is something which is often done
in physics where we have a series expansion and we take a limit of such a
series so that only some diagrams matter (in physics language: renormalization
comes from a lack of understanding in our current system it requires a "UV"
divergence and adding counterterms/regularization, while IR divergences come
from the breakdown of perturbation theory and are treated by resumming
diagrams, the "thermal mass" is an example of such resummation) . If this is
what is happening it might be more helpful to put it in this way: is this a
resummation of the diagrams as eta T-> finite, eta = small ? Could the authors
bring some intuition of why terms with only two edges are dominant? 

### R2 

The paper is based on non-standard tools in machine learning. Even though a
tutorial-like information is given in the supplementary document, I am afraid
that the paper will not be accessible to the general ML audience. That is why I
am suggesting a weak rejection.

I must admit that I am not able to appreciate the paper in details, hence
my confidence.

The paper seems to be well-written, at least grammatically. However, as I
am not familiar with the tools used in the paper, I am not able to judge the
novelty of the approach. My current opinion is that the paper has interesting
ideas, but would not be accessible for the ICML audience. If the other
reviewers have different opinions I would be willing to update my score. 

### R4 

I really struggled to follow the Author's exposition. They crammed a lot of
technical arguments into short paragraphs. They often use terms before defining
them, and have a tendency to use jargon. The Supplementary Material, despite
being 18 pages long, did not help me much. It is also a bit chaotic: the proof
of Corollary 1 is split between Appendices A and B.2.

I am not sure about how relevant the results of this analysis are for
practical applications. In the test on Fashion-MNIST (Figure 5, left), the
Authors ran SGD for approx. 10 datapoints (as explained in Appendix G.3). This
does not seem to me to be a very realistic scenario.

The manuscript contains occasional grammatical errors, and its style should
be improved (see examples) below.

1. Are you confident that a conference paper with 9-page limit is an
appropriate format for the publishing your results? I think your ideas
and work deserve much more space to be clearly explained to the reader who is
not an expert in perturbation calculus and diagram methods.

2. It would be much easier to understand your method if, before describing
the advanced formulation (which perhaps could be published separately in
a journal?), you provided a very simple, step-by-step pedagogical example of
the its application.

3. Could you comment on the applicability of your results to practical use
cases, e.g. to image classifiers trained in realistic settings?

3. I think it would be worth investing some time in improving the wording
of the paper. For example: a) "How does SGD select from among a valley
of minima" - why a valley? did you mean "select from multiple local minima"?
b) "We verify our predictions via precision order-η 3 tests" - why "precision"?
what do you mean here?  c) "Formalism" should not be capitalised in the title
of Sec. 1.1 d) "mean- l distribution over smooth functions l n" (Sec. 2.1) -
what does "mean-l" mean?  e) "For instance, simplification 3 is licensed
because" (Definition 1) - I think it should be "is allowed" f) "root r
participates in no fuzzy edge" (Theorem 1) - I think "root r does not
participate in a fuzzy edge" would be more idiomatic English

4. Is it justified to call your diagrams "Feynman diagrams"? The latter
have a more complicated structure (e.g. loops) than yours.

5. What is the parameter T? How is it related to the number of epochs and
batch size? If I train a CIFAR-10 model for 50 epochs, what is my T?

6. The paper contains many opaque and briefly explained claims. E.g. in
Section 4.4 you write "Intuitively, the presence of the term [diagram]
in our test loss expansion indicates that SGD descends on a covariance-
smoothed landscape." Maybe the problem is on my side, but this is not intuitive
to me at all. Same for "The resulting Stabilized TIC (STIC) uses the metric
implicit in gradient descent to threshold flat from sharp minima" - what does
it mean?

