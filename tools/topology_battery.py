"""Phase 10, Task 1: transport-network design. Does MYCELIA's grown topology
balance cost, efficiency, and robustness better than classic graph designs?

Terminals = inoculum + N food patches. Methods connect the same terminals:
  mycelia        grown from the inoculum on a low-nutrient field with patches
  mst            Euclidean minimum spanning tree (cheapest possible tree)
  complete       straight edge between every terminal pair (most efficient)
Metrics (Bebber et al. 2007 style):
  cost           total edge length (MYCELIA: all living cords, and backbone only)
  efficiency     mean over terminal pairs of euclid / network distance (1 = ideal)
  robustness     fraction of terminal pairs still connected after removing 10%
                 of edges, random and targeted (highest-load edges first)
"""
from __future__ import annotations
import argparse
import heapq
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from mycelia import Config, Environment, Mycelium


def dijkstra(adj, source):
    dist = {source: 0.0}; prev = {}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        for v, w, eid in adj.get(u, ()):
            nd = d+w
            if nd < dist.get(v, math.inf):
                dist[v] = nd; prev[v] = (u, eid)
                heapq.heappush(heap, (nd, v))
    return dist, prev


def build_adj(edges, removed=frozenset()):
    adj = {}
    for eid, (a, b, w) in edges.items():
        if eid in removed:
            continue
        adj.setdefault(a, []).append((b, w, eid)); adj.setdefault(b, []).append((a, w, eid))
    return adj


def contract_chains(edges, keep):
    """Collapse chains of degree-2 nodes into single junction-to-junction
    cords (summed length), as fungal network studies do, so damage removes
    whole cords rather than tiny septated segments. `keep` nodes (terminals)
    are never collapsed."""
    edges = dict(edges)
    adj = {}
    for eid, (a, b, _) in edges.items():
        adj.setdefault(a, set()).add(eid); adj.setdefault(b, set()).add(eid)
    next_id = max(edges, default=-1)+1
    changed = True
    while changed:
        changed = False
        for node in list(adj):
            inc = adj.get(node)
            if node in keep or inc is None or len(inc) != 2:
                continue
            e1, e2 = tuple(inc)
            a1, b1, w1 = edges[e1]; a2, b2, w2 = edges[e2]
            u = b1 if a1 == node else a1
            v = b2 if a2 == node else a2
            if u == v or u == node or v == node:
                continue
            for e, end in ((e1, u), (e2, v)):
                adj[end].discard(e); del edges[e]
            del adj[node]
            edges[next_id] = (u, v, w1+w2)
            adj[u].add(next_id); adj[v].add(next_id); next_id += 1
            changed = True
    return edges


def score_graph(edges, coords, terminal_nodes, rng, damage=0.10):
    """edges: eid -> (a, b, length). coords: node -> (x, y). terminal_nodes:
    list of graph node ids (None if a terminal is unreached)."""
    adj = build_adj(edges)
    pairs = list(itertools.combinations(range(len(terminal_nodes)), 2))
    effs, load, backbone = [], {}, set()
    for i, j in pairs:
        a, b = terminal_nodes[i], terminal_nodes[j]
        if a is None or b is None:
            effs.append(0.0); continue
        dist, prev = dijkstra(adj, a)
        if b not in dist:
            effs.append(0.0); continue
        euclid = math.dist(coords[a], coords[b])
        effs.append(min(1.0, euclid/dist[b]) if dist[b] > 0 else 1.0)
        node = b
        while node != a:
            node, eid = prev[node]
            load[eid] = load.get(eid, 0)+1; backbone.add(eid)

    connected_pairs = [(i, j) for (i, j), e in zip(pairs, effs) if e > 0]

    def connected_fraction(removed):
        # Fraction of pairs that were connected BEFORE damage and remain so.
        if not connected_pairs:
            return 0.0
        adj_r = build_adj(edges, frozenset(removed))
        ok = sum(terminal_nodes[j] in dijkstra(adj_r, terminal_nodes[i])[0]
                 for i, j in connected_pairs)
        return ok/len(connected_pairs)

    k = max(1, int(round(damage*len(edges))))
    eids = list(edges)
    random_removed = rng.choice(eids, size=min(k, len(eids)), replace=False).tolist() if eids else []
    targeted = sorted(eids, key=lambda e: -load.get(e, 0))[:k]
    return {
        'edges': len(edges),
        'cost_total_length': float(sum(w for _, _, w in edges.values())),
        'cost_backbone_length': float(sum(edges[e][2] for e in backbone)),
        'efficiency': float(np.mean(effs)),
        'efficiency_connected_pairs': float(np.mean([e for e in effs if e > 0])) if any(effs) else 0.0,
        'terminals_reached': int(sum(t is not None for t in terminal_nodes)),
        'robust_random_10pct': connected_fraction(random_removed),
        'robust_targeted_10pct': connected_fraction(targeted),
    }


def baseline_graphs(points):
    n = len(points)
    coords = {i: points[i] for i in range(n)}
    complete = {eid: (a, b, math.dist(points[a], points[b]))
                for eid, (a, b) in enumerate(itertools.combinations(range(n), 2))}
    # Prim's MST
    in_tree, mst, eid = {0}, {}, 0
    while len(in_tree) < n:
        a, b = min(((u, v) for u in in_tree for v in range(n) if v not in in_tree),
                   key=lambda uv: math.dist(points[uv[0]], points[uv[1]]))
        mst[eid] = (a, b, math.dist(points[a], points[b])); eid += 1; in_tree.add(b)
    return coords, {'mst': mst, 'complete': complete}


def grow_mycelia(points, size, steps, seed, patch_radius, max_nodes, background=0.8, strength=3.0):
    field = np.full((size, size), background)
    env = Environment(field)
    for (x, y) in points[1:]:
        env.add_patch(x, y, radius=patch_radius, strength=strength)
    cfg = Config(max_nodes=max_nodes, max_tips=48)
    org = Mycelium(env, cfg, seed=seed, position=points[0])
    org.run(steps)
    alive = {nid: n for nid, n in org.nodes.items() if n.alive}
    coords = {nid: (n.x, n.y) for nid, n in alive.items()}
    edges = {s.id: (s.a, s.b, math.dist(coords[s.a], coords[s.b]))
             for s in org.segments.values() if s.alive and s.a in alive and s.b in alive}
    terminals = []
    for (x, y) in points:
        best = min(coords, key=lambda nid: math.dist(coords[nid], (x, y)), default=None)
        terminals.append(best if best is not None and math.dist(coords[best], (x, y)) <= patch_radius else None)
    summary = org.summary()
    return edges, coords, terminals, {k: summary[k] for k in ('carbon_error', 'water_error', 'energy_error')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layouts', type=int, default=5)
    p.add_argument('--sources', type=int, default=5)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--max-nodes', type=int, default=1500)
    p.add_argument('--patch-radius', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--background', type=float, default=0.8)
    p.add_argument('--strength', type=float, default=3.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout in range(a.layouts):
        rng = np.random.default_rng(a.seed+layout)
        c = a.size/2
        points = [(c, c)]
        while len(points) < a.sources+1:
            x, y = rng.uniform(8, a.size-8, 2)
            if all(math.dist((x, y), q) > 12 for q in points):
                points.append((float(x), float(y)))
        edges, coords, terminals, budgets = grow_mycelia(points, a.size, a.steps, a.seed+layout,
                                                         a.patch_radius, a.max_nodes,
                                                         a.background, a.strength)
        result = {'layout': layout, 'points': points, 'mycelia_budget_errors': budgets,
                  'mycelia': score_graph(contract_chains(edges, {t for t in terminals if t is not None}),
                                         coords, terminals, rng)}
        bcoords, base = baseline_graphs(points)
        for name, g in base.items():
            result[name] = score_graph(g, bcoords, list(range(len(points))), rng)
        rows.append(result)
        print(json.dumps({'layout': layout, **{m: {k: round(v, 3) if isinstance(v, float) else v
              for k, v in result[m].items()} for m in ('mycelia', 'mst', 'complete')}}), flush=True)
    methods = ('mycelia', 'mst', 'complete')
    keys = ('cost_total_length', 'cost_backbone_length', 'efficiency', 'efficiency_connected_pairs', 'edges',
            'robust_random_10pct', 'robust_targeted_10pct', 'terminals_reached')
    means = {m: {k: float(np.mean([r[m][k] for r in rows])) for k in keys} for m in methods}
    (a.output/'summary.json').write_text(json.dumps({'schema': 'mycelia.topology-battery.task1.v1',
        'scope': __doc__.strip(), 'args': {k: str(v) for k, v in vars(a).items()},
        'layouts': rows, 'means': means}, indent=2)+'\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    main()
