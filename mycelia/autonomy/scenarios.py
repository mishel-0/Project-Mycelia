"""Scenario families. Each car practises in its own family; evaluation adds
held-out families, including ones where some transferred rules must NOT apply
(e.g. a one-lane road: changing lanes around a block is impossible)."""
from __future__ import annotations
import random
from .sim import Scenario, Hazard


def _block(rng, lanes):
    lane = rng.randrange(lanes); return [Hazard('block', lane, rng.randint(7, 14))], lane


def _ped(rng):
    return Hazard('pedestrian', 0, rng.randint(6, 12), t0=rng.randint(2, 8), duration=rng.randint(3, 6))


def _veh(rng, lane):
    return Hazard('vehicle', lane, rng.randint(4, 9))


def make(family, seed):
    rng = random.Random(f'{family}:{seed}')
    if family == 'city_blocks':        # clear, dry, 2-3 lanes, one blocked lane
        lanes = rng.choice([2, 3]); hz, lane = _block(rng, lanes)
        return Scenario(f'{family}-{seed}', lanes, 'clear', 'dry', hz, start_lane=lane, family=family)
    if family == 'rain_pedestrians':   # wet, clear, pedestrians crossing
        lanes = rng.choice([1, 2])
        return Scenario(f'{family}-{seed}', lanes, 'clear', 'wet', [_ped(rng)], family=family)
    if family == 'fog_traffic':        # fog, slow vehicle ahead, sometimes another in the next lane
        hz = [_veh(rng, 0)] + ([_veh(rng, 1)] if rng.random() < .5 else [])
        return Scenario(f'{family}-{seed}', 2, 'fog', 'dry', hz, family=family)
    if family == 'highway_traffic':    # clear, 3 lanes, slow vehicles
        start = rng.randrange(3); hz = [_veh(rng, start)] + ([_veh(rng, rng.choice([l for l in range(3) if l != start]))] if rng.random() < .5 else [])
        return Scenario(f'{family}-{seed}', 3, 'clear', 'dry', hz, start_lane=start, family=family)
    # Held-out families (evaluation only)
    if family == 'narrow_blocked':     # 1 lane, block: no lane change possible; stopping is the only safe outcome
        return Scenario(f'{family}-{seed}', 1, 'clear', 'dry', [Hazard('block', 0, rng.randint(7, 14))], family=family)
    if family == 'fog_blocks':         # fog + block, side lanes unknown
        lanes = 2; hz, lane = _block(rng, lanes)
        if rng.random() < .5:
            hz.append(Hazard('vehicle', 1 - lane, rng.randint(3, 8)))
        return Scenario(f'{family}-{seed}', lanes, 'fog', 'dry', hz, start_lane=lane, family=family)
    if family == 'rain_traffic':       # wet + slow vehicles
        lanes = rng.choice([2, 3]); start = rng.randrange(lanes)
        return Scenario(f'{family}-{seed}', lanes, 'clear', 'wet', [_veh(rng, start)], start_lane=start, family=family)
    if family == 'mixed':              # random combination of conditions and hazards
        lanes = rng.choice([1, 2, 3]); vis = rng.choice(['clear', 'fog']); surf = rng.choice(['dry', 'wet'])
        start = rng.randrange(lanes); hz = []
        if rng.random() < .6:
            hz.append(Hazard('block', start, rng.randint(8, 14)))
        if rng.random() < .5:
            hz.append(_ped(rng))
        if rng.random() < .5 and lanes > 1:
            hz.append(_veh(rng, rng.randrange(lanes)))
        return Scenario(f'{family}-{seed}', lanes, vis, surf, hz, start_lane=start, family=family)
    raise ValueError(family)


TRAIN_FAMILIES = {'car1': 'city_blocks', 'car2': 'rain_pedestrians', 'car3': 'fog_traffic', 'car4': 'highway_traffic'}
HELD_OUT = ['narrow_blocked', 'fog_blocks', 'rain_traffic', 'mixed']
