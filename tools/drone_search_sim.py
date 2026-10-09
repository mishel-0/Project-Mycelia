"""Symbiotic search simulation: MYCELIA, a classical planner, and both together.

A 2D map hides clustered targets (e.g. people, hotspots, diseased plants).
Every strategy gets the SAME total travel budget (summed flight distance for
drones; summed new hyphal length for MYCELIA) and a sensor radius. Score =
targets found as a function of distance spent.

Strategies:
  lawnmower   D drones sweep parallel lanes (standard search pattern)
  random      D drones, correlated random walk
  planner     D drones, greedy frontier exploration (go to nearest unseen cell)
  mycelia     the real Mycelium organism grows blind; a target only becomes
              food (nutrient patch) once a living compartment senses it
  symbiosis   mycelia + planner: every few steps the planner places a broad
              lure at the centroid of the largest unexplored region, steering
              growth for coverage while MYCELIA exploits found clusters
Simulation only; no claims about real flight control.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from mycelia import Config, Environment, Mycelium

CHECKPOINTS = (200, 400, 800, 1200, 1600)
KINDS = ('lawnmower', 'random', 'planner', 'mycelia', 'mycelia_reward', 'mycelia_reward_punish')


def make_targets(size, k, clusters, rng):
    centers = rng.uniform(10, size-10, size=(clusters, 2))
    pts = []
    while len(pts) < k:
        c = centers[rng.integers(clusters)]
        p = c + rng.normal(0, 4, 2)
        if 2 <= p[0] < size-2 and 2 <= p[1] < size-2:
            pts.append((float(p[0]), float(p[1])))
    return pts


class Scorer:
    def __init__(self, targets, sensor, size):
        self.targets, self.sensor, self.size = targets, sensor, size
        self.found = [None]*len(targets)
        self.seen = np.zeros((size, size), bool)
        yy, xx = np.indices((size, size))
        self._yy, self._xx = yy, xx

    def sense(self, x, y, distance):
        new = []
        for i, (tx, ty) in enumerate(self.targets):
            if self.found[i] is None and math.hypot(tx-x, ty-y) <= self.sensor:
                self.found[i] = distance; new.append(i)
        self.seen |= (self._xx-x)**2+(self._yy-y)**2 <= self.sensor**2
        return new

    def curve(self):
        return {str(b): sum(f is not None and f <= b for f in self.found) for b in CHECKPOINTS}


def frontier_target(seen, x, y):
    unseen = np.argwhere(~seen)
    if not len(unseen):
        return None
    d = (unseen[:, 1]-x)**2+(unseen[:, 0]-y)**2
    yy, xx = unseen[int(np.argmin(d))]
    return float(xx), float(yy)


def run_drones(kind, targets, size, drones, sensor, budget, rng):
    sc = Scorer(targets, sensor, size)
    c = size/2
    pos = [[c, c] for _ in range(drones)]
    heading = [rng.uniform(0, 2*math.pi) for _ in range(drones)]
    # lawnmower lane waypoints per drone (vertical strips)
    lanes = []
    if kind == 'lawnmower':
        strip = size/drones; step = 2*sensor
        for d in range(drones):
            wps, x0, flip = [], d*strip+sensor, False
            x = x0
            while x < (d+1)*strip:
                ys = (size-1, 0) if flip else (0, size-1)
                wps += [(x, ys[0]), (x, ys[1])]; x += step; flip = not flip
            lanes.append(wps)
    wp_index = [0]*drones
    distance = 0.0
    while distance < budget:
        for d in range(drones):
            x, y = pos[d]
            if kind == 'random':
                heading[d] += rng.normal(0, .4)
                nx, ny = x+math.cos(heading[d]), y+math.sin(heading[d])
                if not (0 <= nx < size and 0 <= ny < size):
                    heading[d] += math.pi; continue
            else:
                if kind == 'lawnmower':
                    if wp_index[d] >= len(lanes[d]):
                        continue
                    tx, ty = lanes[d][wp_index[d]]
                else:
                    t = frontier_target(sc.seen, x, y)
                    if t is None:
                        continue
                    tx, ty = t
                dx, dy = tx-x, ty-y; dist = math.hypot(dx, dy)
                if dist < 1e-9:
                    wp_index[d] += 1; continue
                s = min(1.0, dist); nx, ny = x+dx/dist*s, y+dy/dist*s
                if kind == 'lawnmower' and s == dist:
                    wp_index[d] += 1
            distance += math.hypot(nx-x, ny-y); pos[d] = [nx, ny]
            sc.sense(nx, ny, distance)
            if distance >= budget:
                break
        else:
            if kind != 'random' and all((kind == 'lawnmower' and wp_index[d] >= len(lanes[d])) or
                                        (kind == 'planner' and frontier_target(sc.seen, *pos[d]) is None)
                                        for d in range(drones)):
                break
    return sc


def run_mycelia(targets, size, sensor, budget, seed, reward, punish, max_steps, punish_factor=0.6):
    env = Environment(np.full((size, size), 0.8))
    org = Mycelium(env, Config(max_nodes=4000, max_tips=48), seed=seed,
                   position=(size/2, size/2))
    sc = Scorer(targets, sensor, size)
    sc.sense(size/2, size/2, 0.0)
    distance, seen_events, lures = 0.0, 0, 0
    for step in range(max_steps):
        org.step()
        new_ext = org.counts['extension']-seen_events
        seen_events = org.counts['extension']
        if new_ext:
            recent = [e for e in list(org.events)[-new_ext*4:] if e['type'] == 'extension'][-new_ext:]
            for e in recent:
                distance += e['length']
                n = org.nodes[e['node']]
                new = sc.sense(n.x, n.y, distance)
                for i in new:
                    if reward:  # REWARD: a found target becomes real food here
                        tx, ty = targets[i]
                        org.add_resource_patch(tx, ty, radius=4, strength=3)
                if punish and not new and not any(
                        f is not None and math.hypot(targets[j][0]-n.x, targets[j][1]-n.y) <= 3*sensor
                        for j, f in enumerate(sc.found)):
                    # PUNISH: sensed an empty area -> partially drain nutrient there
                    # (ledgered as carbon removed) so tips starve and get pruned.
                    mask = (sc._xx-n.x)**2+(sc._yy-n.y)**2 <= sensor**2
                    removed = float(env.nutrient[mask].sum()*(1-punish_factor))
                    env.nutrient[mask] *= punish_factor
                    org.carbon_removed += removed; lures += 1
        if distance >= budget:
            break
    s = org.summary()
    return sc, {'steps': step+1, 'distance': distance, 'punishments': lures,
                'budget_errors': {k: s[k] for k in ('carbon_error', 'water_error', 'energy_error')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maps', type=int, default=5)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--targets', type=int, default=8)
    p.add_argument('--clusters', type=int, default=3)
    p.add_argument('--drones', type=int, default=4)
    p.add_argument('--sensor', type=float, default=3.0)
    p.add_argument('--budget', type=float, default=1600)
    p.add_argument('--max-steps', type=int, default=4000)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in range(a.maps):
        rng = np.random.default_rng(a.seed+m)
        targets = make_targets(a.size, a.targets, a.clusters, rng)
        row = {'map': m, 'targets': targets}
        for kind in ('lawnmower', 'random', 'planner'):
            row[kind] = run_drones(kind, targets, a.size, a.drones, a.sensor, a.budget,
                                   np.random.default_rng(a.seed+100+m)).curve()
        for kind, rw, pn in (('mycelia', False, False), ('mycelia_reward', True, False),
                             ('mycelia_reward_punish', True, True)):
            sc, info = run_mycelia(targets, a.size, a.sensor, a.budget, a.seed+m, rw, pn, a.max_steps)
            row[kind] = sc.curve(); row[kind+'_info'] = info
        rows.append(row)
        print(json.dumps({'map': m, **{k: row[k]['1600'] for k in KINDS},
                          **{k+'_dist': round(row[k+'_info']['distance']) for k in KINDS[3:]}}), flush=True)
    means = {k: {b: float(np.mean([r[k][str(b)] for r in rows])) for b in CHECKPOINTS} for k in KINDS}
    (a.output/'summary.json').write_text(json.dumps({'schema': 'mycelia.drone-search-sim.v1',
        'scope': __doc__.strip(), 'args': {k: str(v) for k, v in vars(a).items()},
        'maps': rows, 'mean_targets_found_by_distance': means}, indent=2)+'\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
