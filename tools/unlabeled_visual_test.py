"""Feed MRI images without labels; measure cue-addressed reconstruction/retention.

Directory names and filenames stay in the audit manifest and never enter learn().
This is not tumor classification, a clinical accuracy test, or an autonomous image
generator: an input cue chooses stored local patterns for reconstruction.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

# Make the CLI work from a source checkout without relying on PYTHONPATH or an
# editable install; pytest and the documented command both invoke this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from mycelia.visual_memory import (VisualConfig, VisualMycelium, prepare_gray,
                                   prepare_gray_fixed, visual_cues)


def write_json(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def state_hash(model):
    return hashlib.sha256(json.dumps(model.state_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_inputs(dataset, preprocessing='foreground'):
    files = sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.omnia'})
    if not files: raise ValueError('no images found')
    preprocess = {'foreground': prepare_gray, 'fixed': prepare_gray_fixed}.get(preprocessing)
    if preprocess is None: raise ValueError("preprocessing must be 'foreground' or 'fixed'")
    records, gray32, gray16 = [], [], []
    for p in files:
        if p.suffix.lower() == '.omnia':
            try:
                from omnia_sdk.container import OmniaContainer
            except ImportError as exc:
                raise RuntimeError('Reading .omnia images requires omnia-sdk; pass --omnia-sdk PATH or install the SDK') from exc
            with OmniaContainer(p) as container:
                if container.num_slices != 1:
                    raise ValueError(f'{p} contains {container.num_slices} tiles; image learning requires one full-image tile')
                tile = container.get_slice(0)
                if container.pixel_codec == 'jpeg':
                    import io
                    rgb = Image.open(io.BytesIO(tile.tobytes())).convert('RGB')
                    rgb.load()
                elif container.pixel_codec == 'zstd':
                    rgb = Image.fromarray(tile).convert('RGB')
                else:
                    raise ValueError(f'{p} uses unsupported OMNIA pixel codec {container.pixel_codec!r}')
        else:
            with Image.open(p) as im:
                rgb = im.convert('RGB')
                rgb.load()
        source = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        a, b = preprocess(rgb, 32), preprocess(rgb, 16)
        records.append(dict(file=str(p.relative_to(dataset)), source_sha256=source,
                            cue32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                            cue16_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
        gray32.append(a); gray16.append(b)
    return records, np.stack(gray32), np.stack(gray16)


def reconstruction(model, cue):
    result, assignment = model.reconstruct(cue)
    gray = result[..., 0]
    if model.config.size != 32:
        gray = np.asarray(Image.fromarray(gray.astype(np.float32), mode='F').resize((32, 32), Image.Resampling.BILINEAR), dtype=float)
    return gray, assignment


def evaluate(model, cues, reference, indices):
    errors, feature_errors, unsupported = [], [], 0
    outputs = []
    for i in indices:
        gray, assignments = reconstruction(model, cues[i])
        outputs.append(gray)
        errors.append(float(np.mean((gray-reference[i])**2)))
        feature_errors.append(float(np.mean((model.reconstruct(cues[i])[0]-cues[i])**2)))
        unsupported += int(np.any(assignments < 0))
    errors = np.asarray(errors)
    blank_mse = float(np.mean((reference[indices]-.5)**2))
    return dict(images=len(indices), mean_gray_mse=float(errors.mean()), gray_rmse=float(np.sqrt(errors.mean())),
                mean_feature_mse=float(np.mean(feature_errors)), blank_untrained_gray_mse=blank_mse,
                gray_mse_reduction_vs_blank=1-float(errors.mean())/max(blank_mse, 1e-12),
                unsupported_images=unsupported), np.stack(outputs), errors


def fit(cues, indices, config, epochs, seed, output, phase):
    model = VisualMycelium(config); rng = np.random.default_rng(seed)
    carbon, energy = 0., 0.
    for epoch in range(epochs):
        for step, i in enumerate(rng.permutation(indices)):
            update = model.learn(cues[i])
            carbon += update['synthesis_carbon']; energy += update['synthesis_energy']
            if (step+1) % 100 == 0 or step+1 == len(indices):
                model.validate()
                progress = dict(phase=phase, epoch=epoch+1, epochs=epochs, image=step+1,
                                images_per_epoch=len(indices), colonies=len(model.colonies),
                                updates=model.updates, suppressed=model.suppressed_updates)
                write_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    model.freeze()
    evidence = model.summary()
    evidence.update(synthesis_carbon=carbon, synthesis_energy=energy)
    return model, evidence


def panels(path, references, reconstructions, errors):
    order = np.argsort(errors)
    selected = [int(order[k]) for k in np.linspace(0, len(order)-1, 6).astype(int)]
    image = Image.new('RGB', (720, 6*160+50), 'white'); draw = ImageDraw.Draw(image)
    draw.text((15, 12), 'Cue-addressed memory: input / remembered patches (no tumor labels)', fill='black')
    for row, j in enumerate(selected):
        y = row*160+45
        for col, arr in enumerate((references[j], reconstructions[j])):
            tile = Image.fromarray(np.rint(np.clip(arr, 0, 1)*255).astype(np.uint8)).resize((144, 144), Image.Resampling.NEAREST).convert('RGB')
            image.paste(tile, (15+col*160, y))
        draw.text((345, y+30), f'Gray RMSE: {np.sqrt(errors[j]):.4f}', fill='black')
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--omnia-sdk', type=Path,
                        help='omnia-sdk checkout path when its package is not installed')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preprocessing', choices=('foreground', 'fixed'), default='fixed',
                        help='foreground crop/contrast normalization or fixed full-frame resize')
    parser.add_argument('--hierarchical-context', action=argparse.BooleanOptionalAction, default=True,
                        help='learn pair-conditioned three-patch predictive motifs')
    parser.add_argument('--probe-train', type=int, default=600)
    parser.add_argument('--probe-validation', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--fix-probe', action='store_true', help='Test capacity and correction threshold after the initial ablation')
    parser.add_argument('--skip-probes', action='store_true')
    parser.add_argument('--max-colonies', type=int, default=128)
    parser.add_argument('--correction-threshold', type=float, default=.001)
    parser.add_argument('--multiscale-features', action=argparse.BooleanOptionalAction, default=True,
                        help='include broad-range visual cues in the default image learner')
    args = parser.parse_args()
    if args.omnia_sdk:
        sys.path.insert(0, str(args.omnia_sdk.resolve()))
    if min(args.probe_train, args.probe_validation, args.epochs) < 1: parser.error('counts must be positive')
    if args.skip_probes and (args.probe_only or args.fix_probe): parser.error('skip-probes cannot be combined with probe-only or fix-probe')
    chosen = VisualConfig(max_colonies=args.max_colonies,
                          correction_threshold=args.correction_threshold,
                          multiscale_features=args.multiscale_features,
                          preprocessing=args.preprocessing,
                          hierarchical_context=args.hierarchical_context)
    args.output.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    records, gray32, gray16 = load_inputs(args.dataset, preprocessing=args.preprocessing)
    with (args.output/'image-manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    detailed = np.stack([visual_cues(gray, multiscale=args.multiscale_features) for gray in gray32])
    basic = np.stack([visual_cues(gray, detailed=False) for gray in gray16])
    # Probe only inside original Training. Deduplicate actual inputs for BOTH
    # encoders, as well as source pixels; no category names are consulted.
    unique, source, hash32, hash16 = [], set(), set(), set()
    for i, r in enumerate(records):
        if Path(r['file']).parts[0] != 'Training': continue
        duplicate = r['source_sha256'] in source or r['cue32_sha256'] in hash32 or r['cue16_sha256'] in hash16
        source.add(r['source_sha256']); hash32.add(r['cue32_sha256']); hash16.add(r['cue16_sha256'])
        if not duplicate: unique.append(i)
    unique = np.array(unique); np.random.default_rng(args.seed).shuffle(unique)
    if not args.skip_probes and len(unique) < args.probe_train+args.probe_validation: raise ValueError('not enough unique Training images')
    train = unique[:args.probe_train] if not args.skip_probes else np.array([], dtype=int)
    validation = unique[args.probe_train:args.probe_train+args.probe_validation] if not args.skip_probes else np.array([], dtype=int)
    configs = [
        ('16px_specialized_error_gated', VisualConfig(size=16, detailed=False)),
        ('32px_fixed_unconditional', VisualConfig(specialization=False, error_gated=False,
                                                  multiscale_features=args.multiscale_features)),
        ('32px_specialized_unconditional', VisualConfig(error_gated=False,
                                                        multiscale_features=args.multiscale_features)),
        ('32px_specialized_error_gated', VisualConfig(multiscale_features=args.multiscale_features)),
    ]
    if args.fix_probe:
        configs = [
            ('128_colonies_correction_004', VisualConfig(correction_threshold=.004,
                                                         multiscale_features=args.multiscale_features)),
            ('256_colonies_correction_001', VisualConfig(max_colonies=256,
                                                         multiscale_features=args.multiscale_features)),
            ('256_colonies_correction_004', VisualConfig(max_colonies=256, correction_threshold=.004,
                                                         multiscale_features=args.multiscale_features)),
        ]
    if args.skip_probes: configs = []
    configs = [(name, replace(config, preprocessing=args.preprocessing,
                              hierarchical_context=args.hierarchical_context)) for name, config in configs]
    result = dict(scope='Label-free image learning. No tumor categories supplied or predicted. Probe uses Training-only unseen images; final fit includes every raw Training and Testing image.',
                  preprocessing=args.preprocessing,
                  hierarchical_context=args.hierarchical_context,
                  limits='Cue-addressed reconstruction, not class accuracy, patient-independent evaluation, or free image generation. Prototype graph hypotheses; dimensional growing Hypha remains separate.',
                  seed=args.seed, images=len(records), probe_train=len(train), probe_validation=len(validation),
                  probe_epochs=1, final_epochs=args.epochs, probes=[], checks={})
    result['checks']['probe_source_disjoint'] = not ({records[i]['source_sha256'] for i in train} & {records[i]['source_sha256'] for i in validation})
    for key in ('cue16_sha256', 'cue32_sha256'):
        result['checks']['probe_'+key+'_disjoint'] = not ({records[i][key] for i in train} & {records[i][key] for i in validation})
    for name, config in configs:
        inputs = detailed if config.detailed else basic
        model, evidence = fit(inputs, train, config, 1, args.seed, args.output, name)
        before = state_hash(model)
        score, _, errors = evaluate(model, inputs, gray32, validation)
        with (args.output/(name+'-reconstruction.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['file', 'gray_mse'])
            writer.writerows((records[i]['file'], error) for i, error in zip(validation, errors))
        result['probes'].append(dict(name=name, config=asdict(config), learning=evidence, unseen_reconstruction=score))
        result['checks'][name+'_read_only'] = before == state_hash(model)
        print(name+' reconstruction '+json.dumps(score), flush=True)
        write_json(args.output/'summary.json', result)
        del model
    if args.probe_only:
        result['scope'] = 'Training-only corrective probes; no Testing images trained or evaluated.'
        result['checks_passed'] = all(result['checks'].values())
        result['elapsed_seconds'] = time.monotonic()-started
        write_json(args.output/'summary.json', result)
        write_json(args.output/'progress.json', dict(phase='complete', checks_passed=result['checks_passed']))
        return
    # Candidate selection stays within the richer specialized error-gated family;
    # the three promised changes remain present. Evidence can reject its benefit.
    result['final_config'] = asdict(chosen)
    result['full_data_protocol'] = 'Fresh model, shuffled unlabeled inputs, all raw images including duplicates. Final retention metrics are on seen images, never held-out class accuracy.'
    model, evidence = fit(detailed, np.arange(len(records)), chosen, args.epochs, args.seed, args.output, 'all-images-unlabeled')
    path = args.output/'unlabeled-visual-memory.json'; model.save(path)
    before = state_hash(model)
    # Evaluate distinct canonical images so repeated source rows do not inflate
    # the identity-recall score.
    unique_visuals = {}
    for i, record in enumerate(records):
        unique_visuals.setdefault(record['cue32_sha256'], i)
    unique_indices = np.asarray(list(unique_visuals.values()), dtype=int)
    sample = np.random.default_rng(args.seed+1).choice(unique_indices, min(256, len(unique_indices)), replace=False)
    score, recalled, errors = evaluate(model, detailed, gray32, sample)
    episode_results = [model.recognize(detailed[i]) for i in sample]
    exact_episode_matches = sum(item['match_type'] == 'exact' and item['identity'] is not None for item in episode_results)
    exact_episode_ids = [item['identity'] for item in episode_results]
    result['full_data'] = dict(learning=evidence, seen_image_reconstruction=score,
                              episode_identity_recall=dict(scope='Exact canonical visual-identity recall on distinct images sampled from the seen set; not unseen-image generalization or diagnosis.',
                                                           queries=len(sample), exact_matches=exact_episode_matches,
                                                           fraction=exact_episode_matches/max(1, len(sample)),
                                                           distinct_identities=len(set(exact_episode_ids)),
                                                           target='256/256 when at least 256 unique images are available'))
    # Source identity lookup is a separate evaluator. It is not stored in, taught
    # to, or used by the graph model. Exact duplicate sources count as the same ID.
    bank = gray32.reshape(len(gray32), -1).astype(np.float32)
    queries = recalled.reshape(len(recalled), -1).astype(np.float32)
    distance = np.maximum(0, np.sum(queries*queries, axis=1)[:, None]+np.sum(bank*bank, axis=1)[None, :]-2*queries@bank.T)
    nearest = np.argmin(distance, axis=1)
    source_correct = [records[i]['source_sha256'] == records[j]['source_sha256'] for i, j in zip(sample, nearest)]
    cue_correct = [records[i]['cue32_sha256'] == records[j]['cue32_sha256'] for i, j in zip(sample, nearest)]
    blank_nearest = int(np.argmin(np.sum((bank-.5)**2, axis=1)))
    blank_identity_fraction = float(np.mean([records[i]['source_sha256'] == records[blank_nearest]['source_sha256'] for i in sample]))
    result['full_data']['source_identity_retention'] = dict(scope='External evaluator retrieves nearest original image to the remembered grayscale reconstruction; sampled seen images. This is not tumor-class accuracy.',
                                                             queries=len(sample), correct=sum(source_correct), fraction=float(np.mean(source_correct)),
                                                             encoded_equivalent_fraction=float(np.mean(cue_correct)),
                                                             untrained_blank_fraction=blank_identity_fraction)
    result['full_data']['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (args.output/'retention.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['source_file', 'nearest_original_file', 'same_source', 'gray_mse'])
        for i, j, correct, error in zip(sample, nearest, source_correct, errors): writer.writerow([records[i]['file'], records[j]['file'], int(correct), error])
    result['checks']['final_read_only'] = before == state_hash(model)
    result['checks']['sampled_seen_episode_identity_recall'] = exact_episode_matches == len(sample)
    result['checks']['sampled_unique_images_have_unique_episode_ids'] = len(set(exact_episode_ids)) == len(sample)
    result['checks']['all_images_fed'] = model.images_seen == len(records)*args.epochs
    restored = VisualMycelium.load(path)
    result['checks']['save_load_reconstruction_identical'] = all(np.array_equal(model.reconstruct(detailed[i])[0], restored.reconstruct(detailed[i])[0]) for i in sample[:16])
    result['checks']['save_load_resume_state_identical'] = model.state_dict() == restored.state_dict()
    del restored
    routes = [(edge, edge.route_trace, edge.radius) for colony in model.colonies for edge in colony.organism.segments.values()]
    for edge, _, _ in routes:
        edge.route_trace = 0
        edge.radius = model.colonies[0].organism.config.radius*.7
    route_score, route_recalled, _ = evaluate(model, detailed, gray32, sample)
    result['full_data']['without_learned_cord_state'] = route_score
    result['full_data']['cord_ablation_changed_reconstructions'] = int(np.count_nonzero(np.any(np.abs(route_recalled-recalled) > 1e-12, axis=(1, 2))))
    for edge, trace, radius in routes:
        edge.route_trace, edge.radius = trace, radius
    result['checks']['cord_ablation_restored_exact_state'] = before == state_hash(model)
    # Actual stored receptor erasure, not a changed prediction flag.
    for colony in model.colonies:
        for node in colony.organism.nodes.values():
            node.receptor_trace = []; node.receptor_variance = []
    model.validate()
    result['checks']['erasing_memory_removes_all_supported_reconstruction'] = all((model.reconstruct(detailed[i])[1] < 0).all() for i in sample[:16])
    result['checks']['all_resource_budgets_within_1e-8'] = max([p['learning']['max_absolute_budget_error'] for p in result['probes']]+[evidence['max_absolute_budget_error']]) < 1e-8
    result['checks']['no_semantic_label_fields_in_checkpoint'] = 'labels' not in model.state_dict()
    result['checks']['no_external_ml_modules_loaded'] = not any(name.split('.')[0] in {'sklearn', 'torch', 'tensorflow'} for name in sys.modules)
    panels(args.output/'remembered-images.png', gray32[sample], recalled, errors)
    result['elapsed_seconds'] = time.monotonic()-started
    result['checks_passed'] = all(result['checks'].values())
    write_json(args.output/'summary.json', result)
    write_json(args.output/'progress.json', dict(phase='complete', checks_passed=result['checks_passed']))
    print(json.dumps(dict(full_data=result['full_data'], checks=result['checks'], elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)
    if not result['checks_passed']: raise RuntimeError('visual learning integrity check failed')


if __name__ == '__main__': main()
