"""A synthetic image-to-bath protocol, not a medical or fungal sensing law.

Labels never enter this adapter. A fresh dimensional hypha is exposed to a
fixed scan of paired 8x8 image intensities as finite external substrate pulses.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .engine import Hypha
from .state import Parameters

GRID = 8
PULSE_MIN = .1
CONCENTRATION_M = .02
OBSERVABLES = ('tip_pressure_MPa', 'tip_osmotic_pressure_MPa', 'tip_ATP_pmol',
               'tip_apical_cargo_pmol', 'permanent_extension_um',
               'flow_toward_tip_pL_min', 'tip_nutrient_pmol', 'total_water_pL')
# Fixed characteristic scales for numerical/control comparisons; not fitted.
SCALES = np.array([.4, .5, .01, .001, .5, .01, .005, .5])
PROTOCOL = dict(version='mycelia-image-bath-v1', grid=GRID, resize='Pillow L then BOX',
                intensity_scale='gray/255; no per-image min-max normalization',
                pairing='row-major adjacent intensities, basal bath then apical bath',
                pulses=GRID*GRID//2, pulse_duration_min=PULSE_MIN,
                concentration_M='0.02 * intensity; amount = concentration * current bath water',
                initial_ATP_pmol=0, initial_cargo_pmol=0,
                fresh_hypha_per_image=True, initial_state_independent_of_pixels_and_labels=True,
                observables=list(OBSERVABLES), external_replacement_budgeted=True,
                scope='Artificial chemical encoding of image pixels. No measured MRI-to-fungal chemistry relation.')


def load_image(path):
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        rgb.load()
        pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        gray = np.asarray(rgb.convert('L').resize((GRID, GRID), Image.Resampling.BOX), dtype=float)/255
    return gray, pixel_hash


def protocol_hash(parameters=None):
    payload = dict(PROTOCOL, parameters=asdict(parameters or Parameters()))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def new_image_hypha(parameters=None):
    h = Hypha(parameters, initial_ATP_pmol=0, initial_cargo_pmol=0)
    for i in range(2):
        h.set_bath_nutrient(i, 0.0)
    return h


def apply_pulses(h, pairs):
    rows=[]
    for pair in pairs:
        for i, value in enumerate(pair):
            h.set_bath_nutrient(i, float(value) * CONCENTRATION_M * h.baths[i].water_pL)
        state = h.step(PULSE_MIN)
        rows.append([float(state[name]) for name in OBSERVABLES])
    return np.asarray(rows, dtype=float)


def encode_image(gray, parameters=None, *, return_model=False):
    gray = np.asarray(gray, dtype=float)
    if gray.shape != (GRID, GRID) or not np.all(np.isfinite(gray)) or np.any(gray < 0) or np.any(gray > 1):
        raise ValueError('gray must be a finite 8x8 image in [0,1]')
    h = new_image_hypha(parameters)
    rows = apply_pulses(h, gray.ravel().reshape(-1,2))
    if not np.all(np.isfinite(rows)):
        raise ArithmeticError('nonfinite physiological response')
    maximum_error = max(abs(value) for row in h.history for value in row['budget_errors'].values())
    result = dict(features=rows.ravel(), maximum_budget_error=maximum_error,
                  limiter_events=h.limiter_events, final=h.summary())
    if return_model:
        result['model'] = h
    return result
