"""Binary MLCP frame and the JSON / raw-float baselines.

Frame: version | sender | type | seq(2) | evidence ref(2) | uncertainty(1) |
codebook id | n tokens | tokens (1 byte each) | CRC32(4). A bad checksum or
malformed frame unpacks to None: corruption becomes a *missing* message,
never silently wrong data.
"""
from __future__ import annotations
import json, struct, zlib
import numpy as np
from . import VERSION

TYPES = {'evidence_update': 1, 'decision': 2, 'abstain_request': 3}
HEADER = struct.Struct('>BBBHHBBB')


def pack(sender, mtype, seq, evidence, uncertainty, codebook_id, tokens):
    body = HEADER.pack(VERSION, sender, TYPES[mtype], seq & 0xFFFF, evidence & 0xFFFF,
                       int(round(min(max(uncertainty, 0), 1) * 255)), codebook_id, len(tokens)) + bytes(int(t) for t in tokens)
    return body + struct.pack('>I', zlib.crc32(body))


def unpack(frame, check=True):
    try:
        body, crc = frame[:-4], struct.unpack('>I', frame[-4:])[0]
        if check and zlib.crc32(body) != crc:
            return None
        v, sender, t, seq, ev, u, cb, n = HEADER.unpack(body[:HEADER.size])
        tokens = list(body[HEADER.size:])
        if v != VERSION or len(tokens) != n or t not in TYPES.values():
            return None
        return {'version': v, 'sender': sender, 'type': {b: a for a, b in TYPES.items()}[t], 'seq': seq,
                'evidence': ev, 'uncertainty': u / 255, 'codebook': cb, 'tokens': tokens}
    except (struct.error, IndexError, KeyError):
        return None


def pack_raw(sender, mtype, seq, evidence, uncertainty, state):
    body = HEADER.pack(VERSION, sender, TYPES[mtype], seq & 0xFFFF, evidence & 0xFFFF,
                       int(round(min(max(uncertainty, 0), 1) * 255)), 0, len(state)) + np.asarray(state, '>f4').tobytes()
    return body + struct.pack('>I', zlib.crc32(body))


def unpack_raw(frame, check=True):
    try:
        body, crc = frame[:-4], struct.unpack('>I', frame[-4:])[0]
        if check and zlib.crc32(body) != crc:
            return None
        n = HEADER.unpack(body[:HEADER.size])[7]; state = np.frombuffer(body[HEADER.size:], '>f4')
        return state.astype(np.float64) if len(state) == n and np.all(np.isfinite(state)) else None
    except (struct.error, ValueError):
        return None


def pack_json(sender, mtype, seq, evidence, uncertainty, state, fields):
    return json.dumps({'v': VERSION, 'sender': sender, 'type': mtype, 'seq': seq, 'evidence': evidence,
                       'uncertainty': round(float(uncertainty), 4),
                       'state': {f: round(float(x), 4) for f, x in zip(fields, state)}}, separators=(',', ':')).encode()


def unpack_json(frame, fields):
    try:
        d = json.loads(frame.decode()); s = np.array([float(d['state'][f]) for f in fields])
        return s if np.all(np.isfinite(s)) else None
    except Exception:
        return None


def flip_bits(frame, rate, rng):
    a = np.frombuffer(frame, np.uint8).copy(); bits = np.unpackbits(a)
    bits ^= (rng.random(len(bits)) < rate).astype(np.uint8)
    return np.packbits(bits).tobytes()
