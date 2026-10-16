# Mycelia Collective Dissent

Disagreement detection and abstention on top of the Symbiosis Engine.
Code: `tools/collective_dissent_mri.py`; results:
`results/collective-dissent/collective-dissent-results.json`.

> Abstention is a research safety signal, not clinical reliability and not a
> substitute for radiologist review.

## Design

Specialists see the same scan through different evidence, so their mistakes
are less correlated than copies of one model:

| Specialist | Evidence | Test accuracy alone |
|---|---|---|
| A Symbiosis Engine | plastic trace memory, compartment responses | 93.7% |
| B Fixed network | kernel memory, same compartment responses | 94.6% |
| C Pixel memory | kernel memory on raw 64px pixels | 91.8% |
| D Known failures | similarity to 448 scans A got wrong in 2-fold cross-validation on Training | n/a |

The coordinator is a logistic model that predicts "A is wrong" from A/B/C
margins, pairwise disagreements, number of distinct answers, failure
similarity and A's confidence. Protocol: specialists trained on 80% of
Training, coordinator and abstention thresholds fitted on the other 20%
(validation), then everything refit on all Training and the 1,467 Testing
scans scored once.

## Results (held-out Testing, A's prediction, 95% Wilson intervals)

| Target coverage | Answered | Abstained | Accuracy on answered | Errors left | Glioma recall on answered |
|---|---|---|---|---|---|
| 100% (no abstention) | 1,467 | 0 | 93.7% (92.3–94.8) | 93 | 81.5% |
| 95% | 1,383 | 84 | 97.4% (96.4–98.1) | 36 | 91.2% |
| **90%** | **1,319** | **148** | **98.1% (97.2–98.7)** | **25** | **93.0%** |
| 85% | 1,263 | 204 | 98.6% (97.8–99.1) | 18 | 94.1% |
| 80% | 1,189 | 278 | 98.6% (97.7–99.1) | 17 | 93.7% |
| 70% | 1,045 | 422 | 98.7% (97.8–99.2) | 14 | 92.9% |

At 90% coverage the coordinator catches 68 of 93 errors before output, and
the answered-case accuracy interval (97.2–98.7%) does not overlap the
no-abstention interval (92.3–94.8%).

A simple rule, "abstain whenever A, B and C do not all agree", flags 98
scans, catches 55 of 93 errors, and is 97.2% accurate on the 1,369 it answers.

## Findings and caveats

- The strongest signals were A disagreeing with B, and A's own margin and
  confidence. The known-failure memory (D) contributed little (small negative
  weight); it is not yet useful.
- Validation was optimistic (99.7% at 90% coverage vs 98.1% on Testing):
  thresholds tuned on 1,081 scans with 41 errors are noisy.
- Abstentions concentrate on glioma (79 of 148 at 90% coverage). Glioma is
  the hard class, so more glioma scans go to review; 7% of answered glioma
  scans are still wrong.
- 25 errors at 90% coverage are answered confidently, so confident output
  is still not proof of correctness.
- One dataset with near-duplicates across splits; an independent test set
  (e.g. RHUH, TCGA-LGG) is needed before trusting the curve.
