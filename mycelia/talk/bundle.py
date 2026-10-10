"""A trained Mycelia diagnosis pipeline saved as one file.

image -> receptor field -> compartment lattice -> per-compartment compression
-> specialists (Symbiosis Engine, fixed network, pixel memory) -> node states
-> MLCP frames -> receiver decision -> English explanation.
Research prototype; not a medical device.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from mycelia.mycelial_network import MycelialNetwork, NetworkConfig
from mycelia.mlcp import envelope as env
from mycelia.mlcp.decoder import explain
from mycelia.mlcp.nodes import node_state, SENDERS, LABELS


def load_gray(path, size=64):
    with Image.open(path) as im:
        return np.asarray(im.convert('L').resize((size, size), Image.Resampling.BOX), dtype=np.float32) / 255


def kernel_scores(t, a, gamma, q):
    return np.exp(gamma * (q @ t.T - 1)) @ a


class MyceliaBundle:
    def __init__(self, **parts):
        self.__dict__.update(parts)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:  # local file built by tools/build_talk_bundle.py
            return cls(**pickle.load(f))

    def compartments(self, gray):
        net = MycelialNetwork(self.field, NetworkConfig())
        s = net.sense(gray)  # compartments x channels
        return np.stack([((s[j] - m) / sd) @ v for j, (m, sd, v) in enumerate(self.compress)])[None]

    def states(self, xc, gray):
        flat = xc.reshape(1, -1).astype(np.float32)
        q = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
        fail = (q @ self.bank.T).max(1)
        fx = self.fixed; px = gray.reshape(1, -1) - gray.mean(); px = px / max(np.linalg.norm(px), 1e-12)
        return {'engine': node_state(self.engine.scores(flat), 'activation', fail),
                'fixed': node_state(kernel_scores(fx['t'], fx['a'], fx['gamma'], q), 'kernel'),
                'pixels': node_state(kernel_scores(self.pixels['t'], self.pixels['a'], self.pixels['gamma'], px), 'kernel')}

    def diagnose(self, path):
        gray = load_gray(path); h = self.states(self.compartments(gray), gray)
        decoded, missing, messages, frames = [], [], {}, {}
        for sid, s in enumerate(SENDERS):
            cb = self.books[s]; tokens = cb.encode(h[s])[0]
            frame = env.pack(sid, 'evidence_update', 0, 0, float(h[s][0, 5]), cb.id, tokens); m = env.unpack(frame)
            state = cb.decode(np.array([m['tokens']]))[0]
            decoded.append(state[None]); missing.append(np.array([False]))
            messages[s] = {'tokens': m['tokens'], 'state': state}; frames[s] = frame.hex()
        leans = np.array([[int(np.argmax(d[0, :4])) for d in decoded]])
        d = self.receiver.decide(decoded, missing, leans); dec = {k: v[0] for k, v in d.items()}
        sentences = explain(messages, dec, self.receiver.threshold)
        return {'image': str(path), 'decision': 'abstain' if dec['abstain'] else LABELS[int(dec['prediction'])],
                'tentative_label': LABELS[int(dec['prediction'])], 'risk': float(dec['risk']),
                'threshold': float(self.receiver.threshold),
                'probabilities': {l: round(float(p), 4) for l, p in zip(LABELS, dec['probabilities'])},
                'messages': {s: {'tokens': messages[s]['tokens'], 'frame_hex': frames[s]} for s in SENDERS},
                'explanation': [o['text'] for o in sentences], 'explanation_fields': [o['field'] for o in sentences],
                'notice': 'Research prototype, not a medical device or diagnosis.'}
