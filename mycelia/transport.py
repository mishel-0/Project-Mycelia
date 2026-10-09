from __future__ import annotations

import math
import numpy as np


def pressure_solve(organism, dt):
    """Implicit elastic hydraulic relaxation on the living compartment graph.

    (C/dt + graph Laplacian(G)) P = (water - rest_volume)/dt.
    C = rest_volume * wall_compliance. A positive diagonal capacitance fixes
    absolute pressure without an artificial source/sink balancing correction.
    Matrix-free preconditioned conjugate gradient avoids a dense n-by-n array.
    The pressure may be negative relative to the relaxed wall; turgor=max(P,0).
    """
    nodes = [n for n in organism.nodes.values() if n.alive]
    if not nodes:
        return {}, 0.0
    index = {n.id:i for i,n in enumerate(nodes)}
    edges = [e for e in organism.segments.values() if e.alive]
    a = np.array([index[e.a] for e in edges], dtype=int)
    b = np.array([index[e.b] for e in edges], dtype=int)
    g = np.array([e.conductance(organism.config.hydraulic_scale) for e in edges])
    capacity = np.array([n.volume*organism.config.wall_compliance/dt for n in nodes])
    rhs = np.array([(n.water-n.volume)/dt for n in nodes])
    diagonal = capacity.copy()
    np.add.at(diagonal,a,g)
    np.add.at(diagonal,b,g)

    def multiply(x):
        result = capacity*x
        q = g*(x[a]-x[b])
        np.add.at(result,a,q)
        np.add.at(result,b,-q)
        return result

    p = np.array([n.pressure for n in nodes])
    residual = rhs-multiply(p)
    z = residual/diagonal
    direction = z.copy()
    rz = float(residual@z)
    tolerance = 1e-11*max(1.0,float(np.linalg.norm(rhs)))
    for _ in range(max(20,len(nodes)*2)):
        if float(np.linalg.norm(residual)) <= tolerance:
            break
        product = multiply(direction)
        denominator = float(direction@product)
        if denominator <= 0:
            raise ArithmeticError("hydraulic pressure matrix lost positive definiteness")
        alpha = rz/denominator
        p += alpha*direction
        residual -= alpha*product
        z = residual/diagonal
        next_rz = float(residual@z)
        direction = z+(next_rz/rz)*direction
        rz = next_rz
    error = float(np.linalg.norm(rhs-multiply(p)))
    if error > tolerance*10:
        raise ArithmeticError(f"pressure solve did not converge: {error}")
    return {n.id:float(p[i]) for i,n in enumerate(nodes)}, error


def transfer(organism, pool, proposals):
    """Conservative simultaneous transfers with a shared donor budget.

    proposals are (edge_id, source_id, target_id, signed_amount). Signed flow
    retains its physical orientation, while donor allocation is nonnegative.
    """
    demand = {}
    normalized = []
    for edge_id,a,b,amount in proposals:
        donor,receiver = (a,b) if amount >= 0 else (b,a)
        amount = abs(amount)
        demand[donor] = demand.get(donor,0.0)+amount
        normalized.append((edge_id,donor,receiver,amount,1 if donor == a else -1))
    factors = {i:min(1.0,getattr(organism.nodes[i],pool)/v) if v>0 else 1.0 for i,v in demand.items()}
    delta = {i:0.0 for i in organism.nodes}
    moved = {}
    for edge_id,donor,receiver,amount,sign in normalized:
        actual = amount*factors[donor]
        delta[donor] -= actual
        delta[receiver] += actual
        moved[edge_id] = actual*sign
    for i,change in delta.items():
        node = organism.nodes[i]
        value = getattr(node,pool)+change
        if value < -1e-10:
            raise ArithmeticError(f"negative {pool} after conservative transfer")
        setattr(node,pool,max(0.0,value))
    return moved


def transport(organism, dt):
    pressures,error = pressure_solve(organism,dt)
    edges = [e for e in organism.segments.values() if e.alive]
    proposals = [(e.id,e.a,e.b,e.conductance(organism.config.hydraulic_scale)*(pressures[e.a]-pressures[e.b])*dt) for e in edges]
    old_water = {i:n.water for i,n in organism.nodes.items()}
    moved = transfer(organism,"water",proposals)
    tip_nodes = {t.node for t in organism.tips.values() if t.active}
    for pool in ("nutrient","osmolyte"):
        solute = []
        for e in edges:
            a,b = organism.nodes[e.a],organism.nodes[e.b]
            ca = getattr(a,pool)/max(old_water[e.a],1e-12)
            cb = getattr(b,pool)/max(old_water[e.b],1e-12)
            water_amount = moved.get(e.id,0.0)
            advected = water_amount*(ca if water_amount >= 0 else cb)
            diffused = organism.config.solute_diffusion*e.pore_open*(ca-cb)*dt/max(e.length,.1)
            # Phenomenological nutrient cargo transport toward active tips.
            # This transports solute, without inventing an extra water source.
            streamed = 0.0
            if pool == "nutrient":
                drive = int(e.b in tip_nodes)-int(e.a in tip_nodes)
                streamed = organism.config.streaming_rate*e.pore_open*drive*dt*(ca if drive>=0 else cb)
            solute.append((e.id,e.a,e.b,advected+diffused+streamed))
        transfer(organism,pool,solute)
    for n in organism.nodes.values():
        if n.alive:
            n.pressure = (n.water/n.volume-1)/organism.config.wall_compliance
    for e in edges:
        e.flow = moved.get(e.id,0.0)/dt
        e.activity += (1-math.exp(-dt/5))*(abs(e.flow)-e.activity)
    organism.pressure_residual = error
