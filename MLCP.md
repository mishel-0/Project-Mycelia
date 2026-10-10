# MLCP v0.1 — Mycelial Latent Communication Protocol

Node-to-node protocol for Mycelia's specialists: each node turns its state
into learned codebook tokens, sends them in a checksummed binary frame, a
coordinator fuses the messages, and an English decoder translates them
sentence by sentence from message fields only.

Code: `mycelia/mlcp/` (`nodes`, `codebook`, `envelope`, `receiver`,
`decoder`), experiment `tools/mlcp_mri.py`, tests `tests/test_mlcp.py`,
results `results/mlcp/mlcp-results.json`, examples `results/mlcp/explanations.md`.

> A compact encoding is not a new form of intelligence. These are measured
> properties of a communication protocol on one dataset.

## Protocol

- **Node state** h: class probabilities (4), margin, entropy, confidence
  (+ failure similarity for the engine), computed from each specialist's scores.
- **Tokens**: product-quantized k-means codebook (K entries, L parts), fitted
  without labels on validation node states; token = nearest codebook entry.
- **Frame**: version, sender, type, sequence, evidence reference, uncertainty
  (1 byte), codebook id, tokens (1 byte each), CRC32. A failed checksum turns
  a frame into a *missing* message, never into wrong data.
- **Receiver**: softmax fusion of decoded messages plus missing flags (trained
  with random sender loss), and a cross-fitted risk model that abstains
  above a validation-chosen threshold (target 90% coverage).
- **English**: one sentence per field (each sender's tokens and meaning,
  agreement/disagreement, decision), each tagged with its source field.

Senders: Symbiosis Engine, fixed kernel network, pixel memory. Specialists
trained on inner Training; codebook, (K, L) choice and receivers on
validation; Testing (1,467 scans) scored once.

## Results (held-out Testing)

Chosen on validation: **K=16, L=1** — one 4-bit token per sender (smallest
message within 0.5 pt of raw-vector validation accuracy).

| Format | Bytes per scan (3 messages) | Accuracy | Answered | Accuracy on answered | Glioma recall |
|---|---|---|---|---|---|
| Raw float32 state | 130.0 | 93.9% (92.6–95.0) | 1,359 | 97.8% (96.9–98.4) | 82.3% |
| JSON | 729.5 | 93.9% | 1,359 | 97.8% | 82.3% |
| **MLCP tokens** | **45.0** | 93.9% (92.5–95.0) | 1,332 | **98.4% (97.6–99.0)** | 82.3% |

Same decisions with 2.9x fewer bytes than raw and 16x fewer than JSON; the
payload itself is 4 bits per sender versus 28-32 bytes. Most of each 15-byte
MLCP frame is header (10 bytes) and CRC (4 bytes).

### Robustness (mean of 3 corruptions)

| Bit-flip rate | Checksum | Raw | JSON | MLCP |
|---|---|---|---|---|
| 0.1% | yes | 91.6% (347 abstained) | 61.2% (1,189 abstained) | **93.5% (168)** |
| 0.1% | no | 92.5% | — | **93.9%** |
| 1% | yes | 27.2% (1,445 abstained) | 20.3% (all) | **68.8% (1,044)** |
| 1% | no | 80.3%, answered 89.7% | — | **90.7%, answered 98.1%** |

Short frames are hit by fewer flips, and a corrupted token still decodes to
a valid codebook state, so MLCP degrades far more gracefully. With the
checksum, corrupt frames become abstentions rather than wrong answers. (JSON
has no checksum; "yes" there means the parser rejected the damage.)

Dropping one node entirely: accuracy 92.5-94.1% for MLCP vs 93.1-94.2% for
raw — both cope; MLCP abstains slightly more.

### New node joins

A Symbiosis Engine grown with a different seed replaced the original engine,
quantizing with the existing codebook; the receiver was not refit:
MLCP 94.0% (answered 98.3%), raw 93.7% (answered 97.7%).

### Interpretability

Mutual information between a sender's token and the true class: 1.83-1.88
bits out of 2.0. Tokens are readable: e.g. engine token 5 = confident
no-tumor (213/213 validation scans were no-tumor), token 10 = confident
pituitary, token 0 = confident meningioma (93 of 94 meningioma).

### English fidelity

All 1,467 test explanations parsed back to exactly the tokens, per-sender
lean and decision they were generated from (1,467 / 1,467). Example:

> Symbiosis Engine sent tokens [13], meaning: leans meningioma (p 0.67), margin 0.43 ...
> Pixel memory sent tokens [11], meaning: leans notumor ...
> Senders disagree ...
> Decision: withhold prediction for expert review.

## Limits

- The "meaning" in English is the codebook entry's state, i.e. what the token
  says on average, not the sender's exact private numbers.
- Glioma recall is unchanged (82.3%): a protocol does not fix a model's
  blind spot; it carries what the nodes know.
- Tokens come from k-means on validation states (no task-trained codebook
  or emergent multi-step dialogue yet); one dataset with near-duplicates.
