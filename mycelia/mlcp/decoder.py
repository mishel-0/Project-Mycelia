"""English translation of MLCP messages, one sentence per message field.

Sentences are generated only from the decoded message and the receiver's
decision; each carries the field it came from, and `parse` recovers those
fields from the English so fidelity can be checked automatically.
"""
from __future__ import annotations
import re
import numpy as np
from .nodes import LABELS

SENDER_NAMES = {'engine': 'Symbiosis Engine', 'fixed': 'Fixed network', 'pixels': 'Pixel memory'}


def lean(state):
    return LABELS[int(np.argmax(state[:4]))]


def explain(messages, decision, threshold):
    """messages: {sender: None | {'tokens': [...], 'state': decoded vector}}."""
    out = []
    for sender, m in messages.items():
        name = SENDER_NAMES[sender]
        if m is None:
            out.append({'field': f'{sender}.missing', 'text': f'{name}: message missing or corrupt.'}); continue
        s = m['state']
        out.append({'field': f'{sender}.tokens', 'text': f'{name} sent tokens {list(m["tokens"])}, meaning: leans {lean(s)} '
                    f'(p {s[:4].max():.2f}), margin {s[4]:.2f}, uncertainty {s[5]:.2f}.'})
    leans = {k: lean(m['state']) for k, m in messages.items() if m is not None}
    if len(set(leans.values())) > 1:
        out.append({'field': 'disagreement', 'text': 'Senders disagree: ' +
                    ', '.join(f'{SENDER_NAMES[k]} leans {v}' for k, v in leans.items()) + '.'})
    elif leans:
        out.append({'field': 'agreement', 'text': f'All present senders lean {next(iter(leans.values()))}.'})
    r = float(decision['risk'])
    if decision['abstain']:
        text = f'Decision: withhold prediction for expert review (risk {r:.2f} above threshold {threshold:.2f}).'
    else:
        text = f'Decision: predict {LABELS[int(decision["prediction"])]} (risk {r:.2f} within threshold {threshold:.2f}).'
    out.append({'field': 'decision', 'text': text})
    return out


def parse(sentences):
    """Recover fields from the English: per-sender tokens and lean, and the decision."""
    names = {v: k for k, v in SENDER_NAMES.items()}; got = {'senders': {}, 'missing': [], 'decision': None}
    for s in sentences:
        m = re.match(r'(.+?) sent tokens \[([\d, ]*)\], meaning: leans (\w+)', s)
        if m:
            got['senders'][names[m.group(1)]] = {'tokens': [int(t) for t in m.group(2).split(',') if t.strip()],
                                                 'lean': m.group(3)}
        m = re.match(r'(.+?): message missing', s)
        if m:
            got['missing'].append(names[m.group(1)])
        if s.startswith('Decision: withhold'):
            got['decision'] = 'abstain'
        m = re.match(r'Decision: predict (\w+)', s)
        if m:
            got['decision'] = m.group(1)
    return got
