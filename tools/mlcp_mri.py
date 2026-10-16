"""MLCP on brain-tumor MRI: does a learned token protocol let Mycelia's nodes
communicate as well as raw vectors or JSON, in fewer bytes, robustly?

Senders: Symbiosis Engine, fixed kernel network, pixel memory (same
specialists as Collective Dissent). Receiver: coordinator that diagnoses and
abstains. Specialists are trained on inner Training; codebooks, receiver and
(K, L) choice use the validation split; Testing is scored once.
"""
from __future__ import annotations
import argparse, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from mycelia.mlcp import envelope as env
from mycelia.mlcp.codebook import Codebook
from mycelia.mlcp.decoder import explain, parse, lean, SENDER_NAMES
from mycelia.mlcp.nodes import node_state, SENDERS, STATE_FIELDS, LABELS
from mycelia.mlcp.receiver import Receiver
from collective_dissent_mri import engine, PixelMemory, failure_bank, fixed_scores
from symbiosis_mri import compress, FixedNetwork
from train_mycelium import load_images, clean_splits, write_json
from receptor_memory_mri import wilson
from PIL import Image

GRID = [(k, l) for k in (4, 16, 64, 256) for l in (1, 2, 4)]
FIELDS = {'engine': STATE_FIELDS + ('failure_similarity',), 'fixed': STATE_FIELDS, 'pixels': STATE_FIELDS}


def node_states(models, bank, xc, xp):
    eng, fixed, pix = models; flat = xc.reshape(len(xc), -1)
    q = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    fail = (q @ bank.T).max(1) if len(bank) else np.zeros(len(q))
    return {'engine': node_state(eng.scores(flat), 'activation', fail), 'fixed': node_state(fixed_scores(fixed, xc), 'kernel'),
            'pixels': node_state(pix.scores(xp), 'kernel')}


def transmit(states, fmt, books=None, flip=0., check=True, rng=None, drop=()):
    """Send every node's state through the chosen format; return decoded states, missing masks, tokens, bytes."""
    out, missing, tokens, nbytes = [], [], {}, 0
    for sid, s in enumerate(SENDERS):
        h = states[s]; dec = np.zeros_like(h); miss = np.zeros(len(h), bool); toks = [None] * len(h)
        codes = books[s].encode(h) if fmt == 'mlcp' else None
        for i in range(len(h)):
            unc = float(h[i, 5])
            if fmt == 'mlcp':
                f = env.pack(sid, 'evidence_update', i, i, unc, books[s].id, codes[i])
            elif fmt == 'raw':
                f = env.pack_raw(sid, 'evidence_update', i, i, unc, h[i])
            else:
                f = env.pack_json(SENDERS[sid], 'evidence_update', i, i, unc, h[i], FIELDS[s])
            nbytes += len(f)
            if flip:
                f = env.flip_bits(f, flip, rng)
            if s in drop:
                miss[i] = True; continue
            if fmt == 'mlcp':
                m = env.unpack(f, check)
                ok = m is not None and len(m['tokens']) == books[s].parts
                if ok:
                    toks[i] = m['tokens']; dec[i] = books[s].decode(np.array([m['tokens']]))[0]
            elif fmt == 'raw':
                v = env.unpack_raw(f, check); ok = v is not None and len(v) == h.shape[1]
                if ok:
                    dec[i] = v
            else:
                v = env.unpack_json(f, FIELDS[s]); ok = v is not None
                if ok:
                    dec[i] = v
            miss[i] = not ok
        out.append(dec); missing.append(miss); tokens[s] = toks
    leans = np.column_stack([np.where(m, -1, d[:, :4].argmax(1)) for d, m in zip(out, missing)])
    return out, missing, leans, tokens, nbytes / len(states['engine'])


def score(dec, y, coverage_note=True):
    p, ab = dec['prediction'], dec['abstain']; k = int((~ab).sum()); c = int((p[~ab] == y[~ab]).sum())
    return {'accuracy': float((p == y).mean()), 'accuracy_wilson_95': wilson(int((p == y).sum()), len(y)),
            'answered': k, 'abstained': int(ab.sum()), 'accuracy_answered': c / max(k, 1),
            'accuracy_answered_wilson_95': wilson(c, k) if k else None,
            'glioma_recall': float((p[y == 0] == 0).mean())}


def mutual_information(a, b):
    n = len(a); pa = Counter(a); pb = Counter(b); pab = Counter(zip(a, b))
    return sum(c / n * math.log2((c / n) / (pa[x] / n * pb[z] / n)) for (x, z), c in pab.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=Path('results/mycelial-network/cache'))
    ap.add_argument('--output', type=Path, default=Path('results/mlcp')); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    records, _ = load_images(a.dataset, 16); clean, _ = clean_splits(records); tr, te = clean['Training'], clean['Testing']
    y = np.array([LABELS.index(r['label']) for r in records]); ytr, yte = y[tr], y[te]
    def px(r):
        with Image.open(a.dataset / r['file']) as im:
            return np.asarray(im.convert('L').resize((64, 64), Image.Resampling.BOX), dtype=np.float32) / 255
    Ptr = np.stack([px(records[i]) for i in tr]); Pfl = Ptr[:, :, ::-1]; Pte = np.stack([px(records[i]) for i in te])
    C = compress(a.cache, 16); Ctr, Cfl, Cte = C['tr'], C['fl'], C['te']; print('features ready', flush=True)

    def build(idx, seed=0):
        xc = np.concatenate([Ctr[idx], Cfl[idx]]); xp = np.concatenate([Ptr[idx], Pfl[idx]]); yy = np.concatenate([ytr[idx], ytr[idx]])
        return (engine(xc, yy, seed), FixedNetwork().fit(xc, yy), PixelMemory().fit(xp, yy)), failure_bank(Ctr[idx], ytr[idx], a.seed)

    rng = np.random.default_rng(a.seed)
    val = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:int(.2 * (ytr == c).sum())] for c in range(4)])
    inner = np.setdiff1d(np.arange(len(tr)), val); yv = ytr[val]
    models, bank = build(inner); Sv = node_states(models, bank, Ctr[val], Ptr[val]); print('validation states', flush=True)

    # Choose (K, L) on validation with 2-fold receivers: smallest message within 0.5 pt of raw.
    folds = np.random.default_rng(2).permutation(len(yv)) % 2
    def cv_accuracy(dec, miss, leans):
        acc = []
        for f in (0, 1):
            tr_, te_ = folds != f, folds == f
            r = Receiver().fit([d[tr_] for d in dec], [m[tr_] for m in miss], yv[tr_], leans[tr_])
            acc.append((r.decide([d[te_] for d in dec], [m[te_] for m in miss], leans[te_])['prediction'] == yv[te_]).mean())
        return float(np.mean(acc))
    selection = {}
    for fmt in ('raw', 'json'):
        dec, miss, leans, _, b = transmit(Sv, fmt); selection[fmt] = {'bytes': b, 'cv_accuracy': cv_accuracy(dec, miss, leans)}
    books_all = {}
    for k, l in GRID:
        books = {s: Codebook(k, l, seed=a.seed).fit(Sv[s]) for s in SENDERS}; books_all[(k, l)] = books
        dec, miss, leans, _, b = transmit(Sv, 'mlcp', books)
        selection[f'mlcp_K{k}_L{l}'] = {'K': k, 'L': l, 'bits_per_sender': books['engine'].bits, 'bytes': b,
                                        'cv_accuracy': cv_accuracy(dec, miss, leans)}
        print(f'K={k} L={l}', selection[f'mlcp_K{k}_L{l}'], flush=True)
    ok = [(v['bits_per_sender'], -v['cv_accuracy'], name) for name, v in selection.items()
          if name.startswith('mlcp') and v['cv_accuracy'] >= selection['raw']['cv_accuracy'] - .005]
    chosen = min(ok)[2] if ok else max((v['cv_accuracy'], n) for n, v in selection.items() if n.startswith('mlcp'))[1]
    K, L = selection[chosen]['K'], selection[chosen]['L']; books = books_all[(K, L)]
    print('chosen', chosen, flush=True)

    receivers = {}
    for fmt in ('raw', 'json', 'mlcp'):
        dec, miss, leans, _, _ = transmit(Sv, fmt, books); receivers[fmt] = Receiver().fit(dec, miss, yv, leans)

    # Interpretability on validation (chosen codebook).
    _, _, _, vtok, _ = transmit(Sv, 'mlcp', books); glossary = {}; mi = {}
    for s in SENDERS:
        keys = [tuple(t) for t in vtok[s]]; mi[s] = {'token_vs_true_class_bits': mutual_information(keys, list(yv)),
                                                      'true_class_entropy_bits': mutual_information(list(yv), list(yv))}
        entries = []
        for key, n in Counter(keys).most_common(8):
            idx = [i for i, t in enumerate(keys) if t == key]; st = books[s].decode(np.array([key]))[0]
            entries.append({'tokens': list(key), 'count': n, 'leans': lean(st), 'p': round(float(st[:4].max()), 3),
                            'margin': round(float(st[4]), 3), 'true_class_mix': dict(Counter(LABELS[c] for c in yv[idx]))})
        glossary[s] = entries

    # Testing: refit specialists on all Training, score once.
    models, bank = build(np.arange(len(tr))); St = node_states(models, bank, Cte, Pte); print('test states', flush=True)
    test = {}
    for fmt in ('raw', 'json', 'mlcp'):
        dec, miss, leans, tok, b = transmit(St, fmt, books); d = receivers[fmt].decide(dec, miss, leans)
        test[fmt] = {'bytes_per_scan': b, **score(d, yte)}
        if fmt == 'mlcp':
            test_decision, test_tokens, test_dec, test_miss = d, tok, dec, miss
    print('test', {f: (round(v['accuracy'], 4), round(v['accuracy_answered'], 4), round(v['bytes_per_scan'], 1)) for f, v in test.items()}, flush=True)

    robustness = []
    for rate in (1e-3, 1e-2):
        for check in (True, False):
            for fmt in ('raw', 'json', 'mlcp'):
                if fmt == 'json' and not check:
                    continue
                rows = []
                for rep in range(3):
                    dec, miss, leans, _, _ = transmit(St, fmt, books, rate, check, np.random.default_rng(50 + rep))
                    rows.append(score(receivers[fmt].decide(dec, miss, leans), yte))
                robustness.append({'bit_flip_rate': rate, 'checksum': check, 'format': fmt,
                                   'accuracy': float(np.mean([r['accuracy'] for r in rows])),
                                   'accuracy_answered': float(np.mean([r['accuracy_answered'] for r in rows])),
                                   'abstained': float(np.mean([r['abstained'] for r in rows]))})
                print(robustness[-1], flush=True)
    for s in SENDERS:
        for fmt in ('raw', 'mlcp'):
            dec, miss, leans, _, _ = transmit(St, fmt, books, drop=(s,))
            robustness.append({'dropped_node': s, 'format': fmt, **score(receivers[fmt].decide(dec, miss, leans), yte)})

    # New node joins: an engine grown with a different seed uses the existing protocol; receiver unchanged.
    new_engine = engine(np.concatenate([Ctr, Cfl]), np.concatenate([ytr, ytr]), seed=7)
    Sn = dict(St); flat = Cte.reshape(len(Cte), -1); q = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    Sn['engine'] = node_state(new_engine.scores(flat), 'activation', (q @ bank.T).max(1))
    generalization = {}
    for fmt in ('raw', 'mlcp'):
        dec, miss, leans, _, _ = transmit(Sn, fmt, books); generalization[fmt] = score(receivers[fmt].decide(dec, miss, leans), yte)

    # English: fidelity on every test scan, examples for 20.
    fidelity_ok = 0; examples = []
    order = list(np.where(test_decision['abstain'] & (yte == 0))[0][:8]) + list(np.where(~test_decision['abstain'])[0][:12])
    for i in range(len(yte)):
        msgs = {s: (None if test_miss[j][i] else {'tokens': test_tokens[s][i], 'state': test_dec[j][i]}) for j, s in enumerate(SENDERS)}
        dec_i = {k: v[i] for k, v in test_decision.items()}
        out = explain(msgs, dec_i, receivers['mlcp'].threshold); back = parse([o['text'] for o in out])
        want_dec = 'abstain' if dec_i['abstain'] else LABELS[int(dec_i['prediction'])]
        good = back['decision'] == want_dec and all(back['senders'][s]['tokens'] == list(msgs[s]['tokens']) and
                                                    back['senders'][s]['lean'] == lean(msgs[s]['state'])
                                                    for s in SENDERS if msgs[s] is not None)
        fidelity_ok += good
        if i in order:
            examples.append((i, yte[i], out))
    lines = ['# MLCP explanations (held-out test scans)', '',
             'Generated only from decoded MLCP messages and the receiver decision.', '']
    for i, truth, out in examples:
        lines += [f'## Scan {i} (true label: {LABELS[truth]})', ''] + [f'- {o["text"]}  `[{o["field"]}]`' for o in out] + ['']
    (a.output / 'explanations.md').write_text('\n'.join(lines))

    write_json(a.output / 'mlcp-results.json', {
        'method': __doc__, 'chosen': chosen, 'selection': selection, 'test': test, 'robustness': robustness,
        'new_node_generalization': generalization, 'glossary': glossary, 'mutual_information': mi,
        'english_fidelity': {'scans': len(yte), 'parsed_back_exactly': fidelity_ok},
        'receiver_threshold': receivers['mlcp'].threshold, 'seconds': time.monotonic() - start})
    print('english fidelity', fidelity_ok, '/', len(yte), flush=True)


if __name__ == '__main__':
    main()
