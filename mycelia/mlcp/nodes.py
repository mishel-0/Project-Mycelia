"""Node state vectors h_i built from a specialist's class scores (no training)."""
from __future__ import annotations
import numpy as np

LABELS = ('glioma', 'meningioma', 'notumor', 'pituitary')
SENDERS = ('engine', 'fixed', 'pixels')
STATE_FIELDS = ('p_glioma', 'p_meningioma', 'p_notumor', 'p_pituitary', 'margin', 'entropy', 'confidence')


def probabilities(scores, kind):
    s = np.asarray(scores, dtype=np.float64)
    if kind == 'activation':  # Symbiosis Engine: summed positive trace activations
        s = np.log(np.maximum(s, 1e-30))
    else:  # kernel memories: +-1 regression scores
        s = 5 * s
    e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def node_state(scores, kind, extra=None):
    """h = [class probabilities (4), margin, entropy, confidence (+ extra)]."""
    p = probabilities(scores, kind); top = np.sort(p, 1)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(1) / np.log(p.shape[1])
    h = np.column_stack([p, top[:, -1] - top[:, -2], entropy, top[:, -1]])
    return h if extra is None else np.column_stack([h, extra])
