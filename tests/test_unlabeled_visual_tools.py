"""End-to-end image ingestion, checkpoint recall, and experiment provenance."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def test_unlabeled_full_ingestion_and_recall_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path/'dataset'
    rng = np.random.default_rng(392)
    # Full unlabeled feeding must also work without original Training folders;
    # skip-probes must not demand an unused validation split.
    for split, count in [('unlabeled-batch-one', 6), ('unlabeled-batch-two', 2)]:
        folder = dataset/split/'arbitrary-folder-name'
        folder.mkdir(parents=True)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (24, 24), dtype=np.uint8)).save(folder/f'{i}.png')
    output = tmp_path/'result'
    subprocess.run([sys.executable, str(root/'tools/unlabeled_visual_test.py'), '--dataset', str(dataset),
                    '--output', str(output), '--skip-probes', '--probe-train', '2', '--probe-validation', '2',
                    '--max-colonies', '3'], cwd=root, check=True, capture_output=True, text=True)
    result = json.loads((output/'summary.json').read_text())
    assert result['checks_passed']
    assert result['full_data']['learning']['images_seen'] == 8
    assert result['full_data']['learning']['patches_seen'] == 128
    assert result['checks']['no_external_ml_modules_loaded']
    assert result['full_data']['seen_image_reconstruction']['unsupported_images'] == 0
    checkpoint = json.loads((output/'unlabeled-visual-memory.json').read_text())
    assert 'labels' not in checkpoint and 'arbitrary-folder-name' not in json.dumps(checkpoint)
    assert checkpoint['config']['multiscale_features'] is True
    assert checkpoint['config']['preprocessing'] == 'fixed'
    assert checkpoint['config']['hierarchical_context'] is True
    recalled = tmp_path/'recall.png'
    run = subprocess.run([sys.executable, str(root/'tools/recall_visual_image.py'),
                          '--model', str(output/'unlabeled-visual-memory.json'),
                          '--image', str(dataset/'unlabeled-batch-two/arbitrary-folder-name/0.png'),
                          '--output', str(recalled)], cwd=root, check=True, capture_output=True, text=True)
    data = json.loads(run.stdout)
    assert data['supported_patches'] == 16 and recalled.is_file()
    assert Image.open(recalled).size == (32, 32)


def test_masked_patch_transfer_uses_disjoint_unlabeled_examples_and_baselines(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dataset = tmp_path/'dataset'
    rng = np.random.default_rng(710)
    for i in range(9):
        partition = 'Training' if i < 4 else 'Testing'
        folder = dataset/partition/f'unused-folder-{i%2}'
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (24, 24), dtype=np.uint8)).save(folder/f'{i}.png')
    output = tmp_path/'masked'
    excluded = tmp_path/'previous.csv'
    excluded.write_text('image\nTesting/unused-folder-0/4.png\n')
    subprocess.run([sys.executable, str(root/'tools/masked_visual_prediction_test.py'),
                    '--dataset', str(dataset), '--output', str(output), '--train-images', '4',
                    '--validation-images', '4', '--exclude-validation-csv', str(excluded),
                    '--max-colonies', '3'],
                   cwd=root, check=True, capture_output=True, text=True)
    report = json.loads((output/'masked-prediction-summary.json').read_text())
    assert report['checkpoint']['round_trip_exact'] is True
    assert (output/'unlabeled-visual-context-memory.json').is_file()
    assert report['model_settings']['novelty_threshold'] == .012
    assert report['model_settings']['motif_merge_threshold'] == 0.0
    assert report['model_settings']['multiscale_features'] is False
    assert report['checkpoint']['schema'] == 'mycelia-visual-7'
    assert report['counts']['train_images'] == 4
    assert report['counts']['validation_images'] == 4
    assert report['counts']['previously_evaluated_paths_excluded'] == 1
    assert 'Testing' in report['protocol']
    assert report['counts']['masked_patch_predictions'] == 64
    assert report['preprocessing'] == 'fixed'
    assert report['counts']['source_overlap'] == report['counts']['encoded_overlap'] == 0
    assert report['labels_used_for_learning'] is False
    assert report['labels_used_for_selection_or_metrics'] is False
    assert report['preprocessing_caveat'].startswith('none: fixed full-frame resize')
    saved = json.loads((output/'unlabeled-visual-context-memory.json').read_text())
    assert saved['config']['preprocessing'] == 'fixed'
    assert report['paired_image_statistics']['images'] == 4
    assert report['paired_image_statistics']['unit'] == 'image; patient-level independence is unknown'
    assert set(report['gray_intensity_patch_metrics']) == {
        'episodic_context_prediction', 'learned_colony_context_prediction',
        'training_spatial_global_mean', 'same_image_adjacent_patch_mean'}
    hierarchical_output = tmp_path/'hierarchical'
    subprocess.run([sys.executable, str(root/'tools/masked_visual_prediction_test.py'),
                    '--dataset', str(dataset), '--output', str(hierarchical_output),
                    '--train-images', '4', '--validation-images', '4',
                    '--exclude-validation-csv', str(excluded), '--max-colonies', '3',
                    '--hierarchical-context'],
                   cwd=root, check=True, capture_output=True, text=True)
    hierarchical = json.loads((hierarchical_output/'masked-prediction-summary.json').read_text())
    saved_hierarchical = json.loads((hierarchical_output/'unlabeled-visual-context-memory.json').read_text())
    assert hierarchical['model_settings']['hierarchical_context'] is True
    assert saved_hierarchical['config']['hierarchical_context'] is True
    assert hierarchical['learning']['context_motif_connections'] > 0
    assert 'learned_hierarchical_context_prediction' in hierarchical['gray_intensity_patch_metrics']
    assert hierarchical['paired_image_statistics']['learned_hierarchical_context_prediction']['images'] == 4
