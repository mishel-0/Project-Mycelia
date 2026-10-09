"""Local constitutive laws and conservative transfers; parameters are uncalibrated."""
from math import pi
import numpy as np
from .state import Parameters


def reaction(cell, p: Parameters, dt, ledger):
    """Reserve -> soluble carbon -> respiration, finite cargo synthesis and delivery."""
    released = cell.reserve_pmol * (-np.expm1(-p.reserve_release_per_min * dt))
    cell.reserve_pmol -= released
    cell.nutrient_pmol += released
    respired = cell.nutrient_pmol * (-np.expm1(-p.metabolism_per_min * dt))
    cell.nutrient_pmol -= respired
    cell.ATP_pmol += p.ATP_yield * respired
    ledger['respired_carbon_pmol'] += respired
    maintenance = min(cell.ATP_pmol, p.maintenance_density * cell.rest_volume_pL * dt)
    cell.ATP_pmol -= maintenance
    ledger['ATP_spent_pmol'] += maintenance
    if not cell.tip:
        return
    synthesized = cell.nutrient_pmol * (-np.expm1(-p.vesicle_synthesis_per_min * dt))
    if p.vesicle_synthesis_ATP:
        synthesized = min(synthesized, cell.ATP_pmol / p.vesicle_synthesis_ATP)
    cell.nutrient_pmol -= synthesized
    cell.cytoplasmic_cargo_pmol += synthesized
    spent = synthesized * p.vesicle_synthesis_ATP
    cell.ATP_pmol -= spent
    ledger['ATP_spent_pmol'] += spent
    delivered = cell.cytoplasmic_cargo_pmol * (-np.expm1(-p.motor_delivery_per_min * dt))
    if p.motor_delivery_ATP:
        delivered = min(delivered, cell.ATP_pmol / p.motor_delivery_ATP)
    cell.cytoplasmic_cargo_pmol -= delivered
    cell.apical_cargo_pmol += delivered
    spent = delivered * p.motor_delivery_ATP
    cell.ATP_pmol -= spent
    ledger['ATP_spent_pmol'] += spent


def wall_expansion(cell, p, dt, ledger):
    """Yielding confined to a fixed apical growth zone, constrained by insertion."""
    if not cell.tip:
        return 0.0
    growth_zone_volume = p.cross_section_pL_um * min(p.growth_zone_um, cell.rest_length(p))
    mechanical = growth_zone_volume * p.wall_extensibility * max(0, cell.pressure(p) - p.wall_yield_MPa) * dt
    carbon_capacity = cell.apical_cargo_pmol * (-np.expm1(-p.exocytosis_per_min * dt))
    carbon = min(mechanical * p.wall_carbon_pmol_pL, carbon_capacity)
    if p.wall_insertion_ATP:
        carbon = min(carbon, cell.ATP_pmol / p.wall_insertion_ATP)
    volume = carbon / p.wall_carbon_pmol_pL
    cell.apical_cargo_pmol -= carbon
    cell.wall_pmol += carbon
    spent = carbon * p.wall_insertion_ATP
    cell.ATP_pmol -= spent
    ledger['ATP_spent_pmol'] += spent
    cell.rest_volume_pL += volume
    # Water is not created or assigned to growth. Expansion changes pressure;
    # membrane/septal flows in the next solve refill the expanding compartment.
    return volume


def _move_water(model, transfers):
    """Simultaneous donor limits shared across all bath and compartment fluxes."""
    cells, baths = model.cells, model.baths
    pools = {('c', i): c.water_pL for i, c in enumerate(cells)}
    pools.update({('b', i): b.water_pL for i, b in enumerate(baths)})
    demand = {key: 0.0 for key in pools}
    for source, dest, amount, edge in transfers:
        demand[source] += amount
    factors = {key: min(1.0, 0.1 * pools[key] / value) if value else 1.0 for key, value in demand.items()}
    # Limit to 10% donor depletion per substep. This is a numerical safeguard,
    # recorded, not a physiological rate law or hidden source.
    if any(value < 1 - 1e-12 for value in factors.values()):
        model.limiter_events += 1
    concentrations = {('c', i): (c.nutrient_pmol / c.water_pL, c.osmolyte_pmol / c.water_pL) for i, c in enumerate(cells)}
    for source, dest, amount, edge in transfers:
        amount *= factors[source]
        donor = cells[source[1]] if source[0] == 'c' else baths[source[1]]
        receiver = cells[dest[1]] if dest[0] == 'c' else baths[dest[1]]
        donor.water_pL -= amount
        receiver.water_pL += amount
        if edge is not None:
            # Only septal transfer advects soluble cargo; membrane water is selective.
            for field, concentration in zip(('nutrient_pmol', 'osmolyte_pmol'), concentrations[source]):
                moved = amount * concentration
                setattr(donor, field, getattr(donor, field) - moved)
                setattr(receiver, field, getattr(receiver, field) + moved)
            edge.last_flow_pL_min = (amount if source[1] == edge.a else -amount) / model._substep


def water_transport(model, dt):
    """Implicit linear elastic pressure, lagged osmotic concentration, finite baths."""
    p, cells = model.parameters, model.cells
    n = len(cells)
    capacity = np.array([c.rest_volume_pL / p.bulk_modulus_MPa for c in cells])
    permeability = np.array([p.membrane_permeability * c.membrane_area(p) for c in cells])
    drive = np.array([c.osmotic_pressure(p) - model.baths[c.bath].osmotic_pressure(p) for c in cells])
    matrix = np.diag(capacity / dt + permeability)
    rhs = np.array([(c.water_pL - c.rest_volume_pL) / dt for c in cells]) + permeability * drive
    for edge in model.septa:
        g = edge.conductance(p)
        a, b = edge.a, edge.b
        matrix[a, a] += g
        matrix[b, b] += g
        matrix[a, b] -= g
        matrix[b, a] -= g
        edge.last_flow_pL_min = 0.0
    # Unilateral wall elasticity: turgor cannot become negative. A deflated
    # compartment has P=0 and water below its relaxed wall volume (shape omitted).
    # Solve the SPD linear complementarity problem by an active set, enforcing
    # P>=0, A P-rhs>=0, P*(A P-rhs)=0.
    free = set(range(n))
    pressures = np.zeros(n)
    for _ in range(8 * n + 8):
        indexes = sorted(free)
        pressures[:] = 0.0
        if indexes:
            pressures[indexes] = np.linalg.solve(matrix[np.ix_(indexes, indexes)], rhs[indexes])
            negative = [i for i in indexes if pressures[i] < -1e-12]
            if negative:
                free.remove(min(negative, key=lambda i: pressures[i]))
                continue
        residual = matrix @ pressures - rhs
        violating = [i for i in range(n) if i not in free and residual[i] < -1e-12]
        if violating:
            free.add(min(violating, key=lambda i: residual[i]))
            continue
        pressures = np.maximum(pressures, 0.0)
        break
    else:
        raise ArithmeticError('unilateral pressure solve did not converge')
    transfers = []
    for i, c in enumerate(cells):
        signed = dt * permeability[i] * (drive[i] - pressures[i])
        bath, cell = ('b', c.bath), ('c', i)
        if signed >= 0:
            transfers.append((bath, cell, float(signed), None))
        else:
            transfers.append((cell, bath, float(-signed), None))
    for edge in model.septa:
        signed = dt * edge.conductance(p) * (pressures[edge.a] - pressures[edge.b])
        a, b = ('c', edge.a), ('c', edge.b)
        transfers.append((a, b, float(signed), edge) if signed >= 0 else (b, a, float(-signed), edge))
    model._substep = dt
    _move_water(model, transfers)


def solute_diffusion(model, dt):
    """Fick diffusion across septal pores with shared donor budgets."""
    p, cells = model.parameters, model.cells
    for field in ('nutrient_pmol', 'osmolyte_pmol'):
        fluxes = []
        demand = [0.0] * len(cells)
        for edge in model.septa:
            a, b = cells[edge.a], cells[edge.b]
            permeability = p.diffusion_um2_min * pi * p.pore_radius_um ** 2 / p.septum_length_um * 0.001 * edge.opening
            signed = dt * permeability * (getattr(a, field) / a.water_pL - getattr(b, field) / b.water_pL)
            source, dest, amount = (edge.a, edge.b, signed) if signed >= 0 else (edge.b, edge.a, -signed)
            fluxes.append((source, dest, amount))
            demand[source] += amount
        old = [getattr(c, field) for c in cells]
        for source, dest, amount in fluxes:
            moved = amount * min(1.0, old[source] / demand[source]) if demand[source] else 0.0
            setattr(cells[source], field, getattr(cells[source], field) - moved)
            setattr(cells[dest], field, getattr(cells[dest], field) + moved)


def nutrient_uptake(model, dt):
    """Michaelis-Menten capacity, collectively bounded by finite bath substrate."""
    p = model.parameters
    wanted, totals = [], [0.0] * len(model.baths)
    for c in model.cells:
        bath = model.baths[c.bath]
        concentration = bath.nutrient_pmol / bath.water_pL
        amount = dt * p.uptake_density * c.membrane_area(p) * concentration / (p.uptake_Km_M + concentration)
        wanted.append(amount)
        totals[c.bath] += amount
    available = [b.nutrient_pmol for b in model.baths]
    for c, amount in zip(model.cells, wanted):
        taken = amount * min(1.0, available[c.bath] / totals[c.bath]) if totals[c.bath] else 0.0
        model.baths[c.bath].nutrient_pmol -= taken
        c.nutrient_pmol += taken
