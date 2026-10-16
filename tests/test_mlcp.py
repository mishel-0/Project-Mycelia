import numpy as np, pytest
from mycelia.mlcp import envelope as env
from mycelia.mlcp.codebook import Codebook
from mycelia.mlcp.decoder import explain, parse, lean
from mycelia.mlcp.nodes import node_state, LABELS, STATE_FIELDS
from mycelia.mlcp.receiver import Receiver


def states(seed, n=400):
    rng = np.random.default_rng(seed); y = rng.integers(0, 4, n)
    s = np.eye(4)[y] + .6 * rng.standard_normal((n, 4))
    return node_state(s, 'kernel'), y


def test_frame_roundtrip_and_crc():
    f = env.pack(2, 'evidence_update', 7, 204, .31, 9, [17, 4, 91, 12])
    m = env.unpack(f)
    assert m['tokens'] == [17, 4, 91, 12] and m['sender'] == 2 and m['evidence'] == 204 and abs(m['uncertainty'] - .31) < .01
    assert len(f) == env.HEADER.size + 4 + 4
    bad = bytearray(f); bad[12] ^= 1
    assert env.unpack(bytes(bad)) is None and env.unpack(f[:5]) is None


def test_bitflips_become_missing_with_crc():
    rng = np.random.default_rng(0); f = env.pack(1, 'decision', 1, 1, .5, 3, [1, 2])
    flipped = [env.flip_bits(f, .05, rng) for _ in range(200)]
    assert all(env.unpack(x) is None for x in flipped if x != f)
    raw = env.pack_raw(1, 'decision', 1, 1, .5, np.arange(7.))
    assert np.allclose(env.unpack_raw(raw), np.arange(7.))


def test_codebook_bits_determinism_and_decode():
    h, _ = states(0); cb = Codebook(16, 2, seed=3).fit(h)
    t = cb.encode(h); assert t.shape == (len(h), 2) and t.max() < 16 and cb.bits == 8
    assert np.array_equal(t, Codebook(16, 2, seed=3).fit(h).encode(h))
    assert np.abs(cb.decode(t) - h).mean() < np.abs(h - h.mean(0)).mean()


def test_receiver_handles_missing_senders():
    (a, y), (b, _) = states(0), states(0)
    leans = np.column_stack([a[:, :4].argmax(1), b[:, :4].argmax(1)])
    miss = [np.zeros(len(y), bool), np.zeros(len(y), bool)]
    r = Receiver().fit([a, b], miss, y, leans)
    full = r.decide([a, b], miss, leans); assert (full['prediction'] == y).mean() > .7
    gone = r.decide([a, b], [np.zeros(len(y), bool), np.ones(len(y), bool)], leans)
    assert (gone['prediction'] == y).mean() > .6 and full['abstain'].mean() < .2


def test_english_is_traceable_and_parses_back():
    s = np.array([.7, .2, .05, .05, .5, .4, .7]); t = np.array([.1, .8, .05, .05, .7, .3, .8])
    msgs = {'engine': {'tokens': [17], 'state': s}, 'fixed': {'tokens': [3, 9], 'state': t}, 'pixels': None}
    out = explain(msgs, {'risk': .62, 'abstain': True, 'prediction': 0}, .3)
    assert all(o['field'] for o in out)
    back = parse([o['text'] for o in out])
    assert back['senders'] == {'engine': {'tokens': [17], 'lean': 'glioma'}, 'fixed': {'tokens': [3, 9], 'lean': 'meningioma'}}
    assert back['missing'] == ['pixels'] and back['decision'] == 'abstain'
    text = ' '.join(o['text'] for o in out)
    assert 'pituitary' not in text and 'notumor' not in text  # no class absent from the message
