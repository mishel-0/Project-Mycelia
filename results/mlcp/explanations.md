# MLCP explanations (held-out test scans)

Generated only from decoded MLCP messages and the receiver decision.

## Scan 0 (true label: glioma)

- Symbiosis Engine sent tokens [13], meaning: leans meningioma (p 0.67), margin 0.43, uncertainty 0.49.  `[engine.tokens]`
- Fixed network sent tokens [0], meaning: leans meningioma (p 1.00), margin 1.00, uncertainty 0.00.  `[fixed.tokens]`
- Pixel memory sent tokens [11], meaning: leans notumor (p 0.87), margin 0.80, uncertainty 0.35.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans meningioma, Pixel memory leans notumor.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.05 above threshold 0.02).  `[decision]`

## Scan 1 (true label: glioma)

- Symbiosis Engine sent tokens [7], meaning: leans glioma (p 0.98), margin 0.95, uncertainty 0.08.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 2 (true label: glioma)

- Symbiosis Engine sent tokens [9], meaning: leans glioma (p 0.89), margin 0.80, uncertainty 0.27.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 3 (true label: glioma)

- Symbiosis Engine sent tokens [9], meaning: leans glioma (p 0.89), margin 0.80, uncertainty 0.27.  `[engine.tokens]`
- Fixed network sent tokens [7], meaning: leans glioma (p 0.96), margin 0.93, uncertainty 0.12.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 4 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [5], meaning: leans notumor (p 0.42), margin 0.31, uncertainty 0.57.  `[fixed.tokens]`
- Pixel memory sent tokens [10], meaning: leans notumor (p 0.32), margin 0.19, uncertainty 0.91.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans notumor, Pixel memory leans notumor.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.56 above threshold 0.02).  `[decision]`

## Scan 5 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [11], meaning: leans meningioma (p 0.95), margin 0.92, uncertainty 0.15.  `[fixed.tokens]`
- Pixel memory sent tokens [7], meaning: leans pituitary (p 0.27), margin 0.05, uncertainty 0.97.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans meningioma, Pixel memory leans pituitary.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.15 above threshold 0.02).  `[decision]`

## Scan 6 (true label: glioma)

- Symbiosis Engine sent tokens [14], meaning: leans glioma (p 0.98), margin 0.96, uncertainty 0.08.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 7 (true label: glioma)

- Symbiosis Engine sent tokens [14], meaning: leans glioma (p 0.98), margin 0.96, uncertainty 0.08.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 8 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [8], meaning: leans meningioma (p 0.78), margin 0.60, uncertainty 0.45.  `[fixed.tokens]`
- Pixel memory sent tokens [2], meaning: leans glioma (p 0.41), margin 0.13, uncertainty 0.75.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans meningioma, Pixel memory leans glioma.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.28 above threshold 0.02).  `[decision]`

## Scan 9 (true label: glioma)

- Symbiosis Engine sent tokens [7], meaning: leans glioma (p 0.98), margin 0.95, uncertainty 0.08.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [8], meaning: leans glioma (p 0.89), margin 0.81, uncertainty 0.30.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 10 (true label: glioma)

- Symbiosis Engine sent tokens [9], meaning: leans glioma (p 0.89), margin 0.80, uncertainty 0.27.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 11 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [5], meaning: leans notumor (p 0.42), margin 0.31, uncertainty 0.57.  `[fixed.tokens]`
- Pixel memory sent tokens [12], meaning: leans meningioma (p 0.63), margin 0.37, uncertainty 0.66.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans notumor, Pixel memory leans meningioma.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.20 above threshold 0.02).  `[decision]`

## Scan 12 (true label: glioma)

- Symbiosis Engine sent tokens [15], meaning: leans glioma (p 0.48), margin 0.18, uncertainty 0.53.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [14], meaning: leans glioma (p 0.73), margin 0.53, uncertainty 0.54.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 13 (true label: glioma)

- Symbiosis Engine sent tokens [7], meaning: leans glioma (p 0.98), margin 0.95, uncertainty 0.08.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [8], meaning: leans glioma (p 0.89), margin 0.81, uncertainty 0.30.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 14 (true label: glioma)

- Symbiosis Engine sent tokens [4], meaning: leans glioma (p 0.78), margin 0.57, uncertainty 0.42.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 15 (true label: glioma)

- Symbiosis Engine sent tokens [9], meaning: leans glioma (p 0.89), margin 0.80, uncertainty 0.27.  `[engine.tokens]`
- Fixed network sent tokens [2], meaning: leans glioma (p 1.00), margin 1.00, uncertainty 0.01.  `[fixed.tokens]`
- Pixel memory sent tokens [4], meaning: leans glioma (p 0.99), margin 0.99, uncertainty 0.02.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 16 (true label: glioma)

- Symbiosis Engine sent tokens [15], meaning: leans glioma (p 0.48), margin 0.18, uncertainty 0.53.  `[engine.tokens]`
- Fixed network sent tokens [14], meaning: leans glioma (p 0.92), margin 0.85, uncertainty 0.22.  `[fixed.tokens]`
- Pixel memory sent tokens [8], meaning: leans glioma (p 0.89), margin 0.81, uncertainty 0.30.  `[pixels.tokens]`
- All present senders lean glioma.  `[agreement]`
- Decision: predict glioma (risk 0.01 within threshold 0.02).  `[decision]`

## Scan 18 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [3], meaning: leans meningioma (p 0.36), margin 0.10, uncertainty 0.64.  `[fixed.tokens]`
- Pixel memory sent tokens [6], meaning: leans notumor (p 0.65), margin 0.46, uncertainty 0.70.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans meningioma, Pixel memory leans notumor.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.67 above threshold 0.02).  `[decision]`

## Scan 24 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [3], meaning: leans meningioma (p 0.36), margin 0.10, uncertainty 0.64.  `[fixed.tokens]`
- Pixel memory sent tokens [10], meaning: leans notumor (p 0.32), margin 0.19, uncertainty 0.91.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans meningioma, Pixel memory leans notumor.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.55 above threshold 0.02).  `[decision]`

## Scan 30 (true label: glioma)

- Symbiosis Engine sent tokens [6], meaning: leans meningioma (p 0.40), margin 0.18, uncertainty 0.69.  `[engine.tokens]`
- Fixed network sent tokens [9], meaning: leans meningioma (p 0.85), margin 0.72, uncertainty 0.36.  `[fixed.tokens]`
- Pixel memory sent tokens [7], meaning: leans pituitary (p 0.27), margin 0.05, uncertainty 0.97.  `[pixels.tokens]`
- Senders disagree: Symbiosis Engine leans meningioma, Fixed network leans meningioma, Pixel memory leans pituitary.  `[disagreement]`
- Decision: withhold prediction for expert review (risk 0.14 above threshold 0.02).  `[decision]`
