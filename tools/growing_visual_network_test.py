"""Grow a real hyphal graph toward unlabeled prediction-error hotspots.

Image prediction error is converted to finite, transient nutrient pulses. This
cross-level coupling is an explicit computational hypothesis, not a known
fungal MRI mechanism. A resource-matched control receives the same pulse sizes
at shuffled locations.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import time

import numpy as np

from mycelia import Config, Environment, Mycelium
from mycelia.predictive_memory import PredictiveMycelium, masked_neighbor_context, patch_grid
from masked_patch_prediction_test import data_from_phase1


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def patch_errors(model, cue):
    patches = patch_grid(cue)
    errors = np.empty(16)
    for position in range(16):
        context = masked_neighbor_context(cue, position)
        prediction, _, _ = model.predict(context, position)
        errors[position] = np.mean((prediction[..., 0] - patches[position][..., 0]) ** 2)
    return errors


def make_network(seed, max_nodes):
    # A viable basal field prevents the inoculum from dying before distant
    # image-cue resources can be reached. Cue pulses remain the spatial signal.
    environment = Environment(np.full((32, 32), 1.0), diffusion=.02)
    config = Config(dt=.5, max_nodes=max_nodes, max_tips=24, branch_interval=2.0,
                    branch_rate=.55, fusion_radius=.85, starvation_time=18.0,
                    growth_speed=.65, reinforcement_rate=2.0)
    return Mycelium(environment, config, seed=seed, position=(15.5, 15.5))


def patch_center(index):
    row, col = divmod(int(index), 4)
    return col * 8 + 3.5, row * 8 + 3.5


def graph_metrics(organism, target_positions, seed):
    nodes = {i: n for i, n in organism.nodes.items() if n.alive}
    edges = [e for e in organism.segments.values() if e.alive and e.a in nodes and e.b in nodes]
    adjacency = {i: [] for i in nodes}
    conductance = {}
    for edge in edges:
        value = max(edge.conductance(organism.config.hydraulic_scale), 1e-12)
        conductance[edge.id] = value
        cost = 1 / value
        adjacency[edge.a].append((edge.b, edge.id, cost))
        adjacency[edge.b].append((edge.a, edge.id, cost))

    def distances(removed=frozenset(), source=0):
        output = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != output[current]:
                continue
            for other, edge_id, cost in adjacency[current]:
                if edge_id in removed:
                    continue
                proposal = distance + cost
                if proposal < output.get(other, float('inf')):
                    output[other] = proposal
                    heapq.heappush(queue, (proposal, other))
        return output

    def connected(removed=frozenset()):
        seen = {0} if 0 in nodes else set()
        queue = list(seen)
        while queue:
            current = queue.pop()
            for other, edge_id, _ in adjacency[current]:
                if edge_id not in removed and other not in seen:
                    seen.add(other); queue.append(other)
        return seen

    def target_efficiency(removed=frozenset()):
        costs = distances(removed)
        result = []
        for x, y in target_positions:
            nearest = min(nodes, key=lambda i: (nodes[i].x-x)**2 + (nodes[i].y-y)**2)
            distance = costs.get(nearest)
            result.append(1 / distance if distance is not None and distance > 0 else 0.0)
        return float(np.mean(result)) if result else 0.0

    intact = distances()
    root_efficiency = float(np.mean([1 / distance for node, distance in intact.items() if node != 0 and distance > 0])) if len(intact) > 1 else 0.0
    root_targets = target_efficiency()
    connected_fraction = len(connected()) / max(1, len(nodes))
    # Unweighted Brandes edge betweenness identifies structurally central links.
    centrality = {edge.id: 0.0 for edge in edges}
    for source in nodes:
        stack, predecessors = [], {v: [] for v in nodes}
        sigma = dict.fromkeys(nodes, 0.0); sigma[source] = 1.0
        distance = dict.fromkeys(nodes, -1); distance[source] = 0
        queue = [source]
        for vertex in queue:
            stack.append(vertex)
            for other, edge_id, _ in adjacency[vertex]:
                if distance[other] < 0:
                    queue.append(other); distance[other] = distance[vertex] + 1
                if distance[other] == distance[vertex] + 1:
                    sigma[other] += sigma[vertex]
                    predecessors[other].append((vertex, edge_id))
        dependency = dict.fromkeys(nodes, 0.0)
        while stack:
            vertex = stack.pop()
            for previous, edge_id in predecessors[vertex]:
                share = sigma[previous] / max(sigma[vertex], 1e-12) * (1 + dependency[vertex])
                dependency[previous] += share
                centrality[edge_id] += share
    attack_count = min(len(edges), max(1, int(np.ceil(.1 * len(edges))))) if edges else 0
    targeted = set(sorted(centrality, key=centrality.get, reverse=True)[:attack_count])
    rng = np.random.default_rng(seed)
    random_attack = set(rng.choice([e.id for e in edges], size=attack_count, replace=False)) if attack_count else set()
    targeted_connected = len(connected(targeted)) / max(1, len(nodes))
    random_connected = len(connected(random_attack)) / max(1, len(nodes))
    targeted_efficiency = target_efficiency(targeted)
    random_efficiency = target_efficiency(random_attack)
    summary = organism.summary()
    return dict(
        living_nodes=len(nodes), living_edges=len(edges),
        active_tips=summary['active_tips'], total_length=summary['total_length'],
        construction_volume=sum(np.pi * edge.radius**2 * edge.length for edge in edges),
        mean_hypha_radius=float(np.mean([edge.radius for edge in edges])) if edges else 0.0,
        reinforced_edges=sum(edge.radius > organism.config.radius*.7+1e-7 for edge in edges),
        reinforcement_radius_added=float(sum(max(0.0,edge.radius-organism.config.radius*.7) for edge in edges)),
        mean_flow=float(np.mean([abs(edge.flow) for edge in edges])) if edges else 0.0,
        branch_events=summary['events'].get('branch', 0), fusion_events=summary['events'].get('fusion', 0),
        retraction_events=summary['events'].get('retraction', 0),
        tip_death_events=summary['events'].get('tip_death', 0),
        tip_reactivation_events=summary['events'].get('tip_reactivation', 0),
        tip_resprout_events=summary['events'].get('tip_resprout', 0),
        root_connected_fraction=connected_fraction, root_to_network_efficiency=root_efficiency,
        root_to_heldout_error_hotspot_efficiency=root_targets,
        damage_edges_removed=attack_count, damage_fraction=attack_count/max(1,len(edges)),
        connected_fraction_after_high_betweenness_damage=targeted_connected,
        connected_fraction_after_random_damage=random_connected,
        heldout_hotspot_efficiency_after_high_betweenness_damage=targeted_efficiency,
        heldout_hotspot_efficiency_after_random_damage=random_efficiency,
        carbon_added=organism.carbon_added, carbon_removed=organism.carbon_removed,
        carbon_uptake=summary['uptake'], carbon_growth_spent=organism.growth_spent,
        carbon_budget_error=summary['carbon_error'], water_budget_error=summary['water_error'],
        energy_budget_error=summary['energy_error'], simulated_time=summary['time'], steps=summary['step'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--phase1-output', type=Path, required=True)
    parser.add_argument('--phase2-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-nodes', type=int, default=192)
    parser.add_argument('--max-train', type=int, help='Optional topology smoke-run cap')
    parser.add_argument('--max-test', type=int, help='Optional held-out metric cap')
    parser.add_argument('--pulse-radius', type=float, default=1.5)
    # Growth operates on substrate carbon amounts, while masked-patch MSE is
    # dimensionless and small. This gain maps error magnitude into meaningful
    # finite resource pulses; the matched control gets the identical amounts.
    parser.add_argument('--pulse-scale', type=float, default=5.0)
    args = parser.parse_args()
    if args.max_nodes < 8 or args.pulse_radius <= 0 or args.pulse_scale <= 0:
        parser.error('invalid node capacity or nutrient-pulse parameters')
    args.output.mkdir(parents=True, exist_ok=True)
    (train_paths, test_paths, _, _, _, train_cues, _, test_cues) = data_from_phase1(
        args.dataset, args.phase1_output, max_train=args.max_train, max_test=args.max_test, seed=args.seed)
    model = PredictiveMycelium.load(args.phase2_output / 'masked-predictive-memory.json')
    if not model.frozen:
        raise ValueError('growth coupling requires a frozen predictive model')
    guided = make_network(args.seed, args.max_nodes)
    control = make_network(args.seed, args.max_nodes)
    prior_guided_positions, prior_control_positions = [], []
    randomizer = np.random.default_rng(args.seed + 19)
    # data_from_phase1 intentionally returns deterministic sorted paths, but
    # those paths are grouped by dataset directory. Shuffle before the online
    # growth stream so filesystem/class-folder ordering cannot shape topology.
    training_order = np.random.default_rng(args.seed + 7).permutation(len(train_cues))
    heldout_targets = []
    started = time.monotonic()

    for image_index, cue_index in enumerate(training_order):
        cue = train_cues[cue_index]
        errors = patch_errors(model, cue)
        hot = np.argsort(errors)[-2:]
        strengths = [.005 + args.pulse_scale * float(errors[index]) for index in hot]
        # Each sample replaces the prior sensory field with transient sources;
        # removals and additions enter the organism's external-carbon ledger.
        for organism, previous in ((guided, prior_guided_positions), (control, prior_control_positions)):
            for position in previous:
                x, y = patch_center(position)
                organism.deplete_region(x, y, args.pulse_radius * 1.5)
        control_positions = randomizer.choice(16, size=len(hot), replace=False).tolist()
        for position, strength, shuffled_position in zip(hot, strengths, control_positions):
            if strength > 0:
                x, y = patch_center(position)
                guided.add_resource_patch(x, y, radius=args.pulse_radius, strength=strength)
                cx, cy = patch_center(shuffled_position)
                control.add_resource_patch(cx, cy, radius=args.pulse_radius, strength=strength)
        prior_guided_positions = [int(p) for p in hot]
        prior_control_positions = control_positions
        guided.step(); control.step()
        if (image_index + 1) % 100 == 0 or image_index + 1 == len(train_cues):
            progress = dict(images_processed=image_index + 1, images_total=len(train_cues),
                            guided_nodes=guided.summary()['stored_nodes'],
                            guided_tips=guided.summary()['active_tips'],
                            guided_branches=guided.counts['branch'], guided_fusions=guided.counts['fusion'],
                            control_nodes=control.summary()['stored_nodes'], elapsed_seconds=time.monotonic()-started)
            write_json(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)

    # Evaluate routes to held-out, high-error visual cues. No held-out image is
    # used to grow either physical network.
    for cue in test_cues:
        for index in np.argsort(patch_errors(model, cue))[-2:]:
            heldout_targets.append(patch_center(index))
    guided_metrics = graph_metrics(guided, heldout_targets, args.seed)
    control_metrics = graph_metrics(control, heldout_targets, args.seed + 1)
    guided.save(args.output / 'guided-growing-network.json')
    control.save(args.output / 'shuffled-control-network.json')
    result = dict(
        phase='3 - growth, transport and damage assay',
        hypothesis='Local masked-patch prediction error is represented as a finite, temporary nutrient pulse that guides growth. Pulse quantity is matched; only locations differ between guided and shuffled control.',
        limits='This coupling is a software hypothesis. It does not mean fungal hyphae process MRI. The underlying dimensional growth, fusion, transport and pruning rules are separate computational mechanisms and are not empirically calibrated to this image-to-nutrient mapping.',
        data=dict(training_images=len(train_paths), heldout_images=len(test_paths),
                  labels_used_for_learning=False, heldout_used_for_growth=False,
                  training_stream='seeded random permutation of clean unlabeled training images',
                  hotspots_per_training_image=2, pulse_radius=args.pulse_radius, pulse_scale=args.pulse_scale),
        parameters=guided.config.as_dict(),
        guided=guided_metrics, shuffled_location_control=control_metrics,
        paired_differences=dict(
            heldout_hotspot_efficiency=guided_metrics['root_to_heldout_error_hotspot_efficiency']-control_metrics['root_to_heldout_error_hotspot_efficiency'],
            random_damage_connected_fraction=guided_metrics['connected_fraction_after_random_damage']-control_metrics['connected_fraction_after_random_damage'],
            targeted_damage_connected_fraction=guided_metrics['connected_fraction_after_high_betweenness_damage']-control_metrics['connected_fraction_after_high_betweenness_damage']),
        checks=dict(
            guided_branched=guided_metrics['branch_events'] > 0,
            guided_fused=guided_metrics['fusion_events'] > 0,
            guided_reinforced=guided_metrics['reinforced_edges'] > 0,
            guided_pruned=guided_metrics['retraction_events'] > 0,
            heldout_hotspot_targets_present=len(heldout_targets) > 0,
            carbon_budgets_conserved=max(abs(guided_metrics['carbon_budget_error']), abs(control_metrics['carbon_budget_error'])) < 1e-7,
            water_budgets_conserved=max(abs(guided_metrics['water_budget_error']), abs(control_metrics['water_budget_error'])) < 1e-7,
            energy_budgets_conserved=max(abs(guided_metrics['energy_budget_error']), abs(control_metrics['energy_budget_error'])) < 1e-7),
        elapsed_seconds=time.monotonic()-started)
    result['checks_passed'] = all(result['checks'].values())
    write_json(args.output / 'summary.json', result)
    write_json(args.output / 'progress.json', dict(phase='complete', checks_passed=result['checks_passed']))
    print(json.dumps(result, indent=2), flush=True)
    if not result['checks_passed']:
        raise SystemExit('growing-network integrity check failed')


if __name__ == '__main__':
    main()
