"""Controlled inoculum transfer and directional-memory measurements.

This isolates a candidate local polarity mechanism. It is a computational assay,
not a claim of experimental replication or biological identification.
"""
from copy import deepcopy
from dataclasses import replace
import math,random
import numpy as np
from .environment import Environment
from .organism import Mycelium


def transfer_inoculum(source,*,nutrient=.6,size=64,seed=1,condition='memory',sham=None):
    """Transfer the root compartment only, removing all old geometry and tips.

    Every arm starts with the trained source's identical physical resource pools.
    The sham arm substitutes only the directional trace from a separately exposed
    uniform-history organism. This matching is an artificial causal control.
    The fresh phase has its own explicit initial budget; no source is mutated.
    """
    allowed=('memory','erased','bias_off','inheritance_off','rotated','sham')
    if condition not in allowed or (condition=='sham' and sham is None):raise ValueError('invalid transfer condition')
    config=replace(source.config,foraging_memory_rate=0.,foraging_memory_gain=0. if condition=='bias_off' else source.config.foraging_memory_gain,foraging_memory_inheritance=0. if condition=='inheritance_off' else source.config.foraging_memory_inheritance)
    env=Environment(np.full((size,size),nutrient),diffusion=config.diffusion)
    center=size//2
    o=Mycelium(env,config,seed=seed,position=(center,center))
    n=deepcopy(source.nodes[0]);n.id=0;n.x=n.y=float(center)
    if not n.alive:raise ValueError('source inoculum is dead')
    if condition=='erased':n.polarity_trace_x=n.polarity_trace_y=0.
    elif condition=='rotated':n.polarity_trace_x*=-1;n.polarity_trace_y*=-1
    elif condition=='sham':
        n.polarity_trace_x=sham.nodes[0].polarity_trace_x;n.polarity_trace_y=sham.nodes[0].polarity_trace_y
    o.nodes={0:n};o.tips={};o.segments={};o.next_node=1;o.next_tip=o.next_segment=0
    # Bud polarity is regenerated independently of the training direction.
    rotation=random.Random(seed+1729).uniform(0,2*math.pi)
    for k in range(min(4,config.max_tips)):
        a=rotation+k*math.pi/2;o._tip(0,math.cos(a),math.sin(a))
    o.initial_carbon=o.total_carbon();o.initial_water=o.total_water();o.initial_energy=n.energy
    o._event('inoculum_transfer',condition=condition,retained_compartments=1,removed_old_nodes=len(source.nodes)-1,matched_resources=True)
    o._validate()
    return o


def direction_metrics(organism,angle):
    """Length-weighted alignment of *new* segments with the old bait direction."""
    if not math.isfinite(angle):raise ValueError('angle must be finite')
    target=(math.cos(angle),math.sin(angle));length=projection=forward=0.
    for e in organism.segments.values():
        if not e.alive:continue
        a,b=organism.nodes[e.a],organism.nodes[e.b]
        d=math.hypot(b.x-a.x,b.y-a.y)
        if d<=0:continue
        alignment=((b.x-a.x)*target[0]+(b.y-a.y)*target[1])/d
        length+=e.length;projection+=e.length*alignment
        if alignment>0:forward+=e.length
    return {'alignment':projection/length if length else None,'forward_length_fraction':forward/length if length else None,'new_length':length,'new_segments':sum(e.alive for e in organism.segments.values())}
