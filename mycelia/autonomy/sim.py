"""Grid road simulator with symbolic observations.

Road of `length` cells and 1-3 lanes. The car advances one cell per tick when
driving, every other tick when slow, not at all when stopped. Hazards:
  block       static, occupies one cell forever
  pedestrian  crosses lanes at a cell between ticks [t0, t0+duration)
  vehicle     slow vehicle in a lane moving one cell every 2 ticks
Visibility: clear (sees 6 cells) or fog (2 cells). Surface: dry, or wet (a stop
issued while moving still rolls one more cell). Episode ends at the finish
(success), on collision, or at the time limit (stuck).
"""
from __future__ import annotations
from dataclasses import dataclass, field

ACTIONS = ('keep', 'left', 'right', 'slow', 'stop')
RANGE = {'clear': 6, 'fog': 2}


@dataclass
class Hazard:
    kind: str
    lane: int
    pos: float
    t0: int = 0
    duration: int = 0

    def occupies(self, lane, pos, t):
        if self.kind == 'pedestrian':
            return self.t0 <= t < self.t0 + self.duration and pos == self.pos
        return lane == self.lane and int(self.pos) == pos

    def step(self, t):
        if self.kind == 'vehicle' and t % 2 == 0:
            self.pos += 1


@dataclass
class Scenario:
    id: str
    lanes: int
    visibility: str
    surface: str
    hazards: list
    start_lane: int = 0
    length: int = 24
    limit: int = 60
    family: str = ''
    extra_roll: int = 0   # hidden environment shift (e.g. worn tyres): extra cells rolled after a stop

    @classmethod
    def from_dict(cls, d):
        return cls(**{**d, 'hazards': [Hazard(**h) for h in d['hazards']]})


def observe(sc, hazards, lane, pos, t):
    """Symbolic features only; anything beyond visibility range is unknown."""
    rng = RANGE[sc.visibility]

    def nearest(l):
        best = None
        for h in hazards:
            for d in range(1, rng + 1):
                if h.occupies(l, pos + d, t) or (h.kind == 'pedestrian' and h.pos == pos + d and h.t0 - 2 <= t < h.t0 + h.duration):
                    if best is None or d < best[1]:
                        best = (h.kind, d)
                    break
        return best

    ahead = nearest(lane)
    def side(l):
        if l < 0 or l >= sc.lanes:
            return 'none'
        if rng < 3:
            return 'unknown'
        busy = any(h.occupies(l, pos + d, t) for h in hazards for d in range(-1, 4)) or any(
            h.kind == 'vehicle' and h.lane == l and pos - 2 <= h.pos <= pos + 3 for h in hazards)
        return 'no' if busy else 'yes'
    return {'ahead': ahead[0] if ahead else 'none',
            'distance': 'none' if not ahead else ('near' if ahead[1] <= 2 else 'mid' if ahead[1] <= 4 else 'far'),
            'left_free': side(lane - 1), 'right_free': side(lane + 1),
            'visibility': sc.visibility, 'surface': sc.surface}


def reflex(obs):
    """Built-in safety reflex (not learned): stop when something is near ahead."""
    return 'stop' if obs['distance'] == 'near' else 'keep'


def simulate(sc, policy, trace=False, brake=0):
    """Run one episode. policy(obs) -> (action, rule_id or None).
    `brake` is the car's hidden capability: extra cells rolled after a stop
    while moving (0 strong, 1 normal, 2 weak). Wet roads and `sc.extra_roll`
    add to it. Neither is visible in observations."""
    hazards = [Hazard(**vars(h)) for h in sc.hazards]
    lane, pos, moving, t, stops, unnecessary, idle, fired = sc.start_lane, 0, True, 0, 0, 0, 0, []
    steps = []
    def done(outcome, ticks):
        out = {'outcome': outcome, 'ticks': ticks, 'stops': stops, 'unnecessary_stops': unnecessary, 'idle': idle, 'fired': fired}
        if trace:
            out['trace'] = steps
        return out
    while t < sc.limit:
        obs = observe(sc, hazards, lane, pos, t)
        action, rule = policy(obs)
        steps.append((obs, action, rule))
        if rule:
            fired.append(rule)
        if action in ('left', 'right'):
            new = lane + (-1 if action == 'left' else 1)
            if 0 <= new < sc.lanes:
                lane = new
        advance = 0
        if action == 'stop':
            stops += 1; unnecessary += obs['distance'] in ('none', 'far')
            advance = (brake + (sc.surface == 'wet') + sc.extra_roll) if moving else 0; moving = False
        elif action == 'slow':
            advance = t % 2; moving = True
        else:
            advance = 1; moving = True
        idle += advance == 0
        for step in range(advance):
            pos += 1
            if any(h.occupies(lane, pos, t) for h in hazards):
                return done('collision', t + 1)
        t += 1
        for h in hazards:
            h.step(t)
        if any(h.occupies(lane, pos, t) for h in hazards):
            return done('collision', t)
        if pos >= sc.length:
            return done('success', t)
    return done('stuck', t)
