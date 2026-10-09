from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict
import json
import heapq
import math
from pathlib import Path
import random

import numpy as np

from .environment import Environment
from .state import Compartment, Config, Segment, Tip
from .transport import transport


class Mycelium:
    """One compartment graph owns the complete physiological timestep.

    Carbon and water budgets include the environment. Metabolic carbon exits
    through an explicit respiration ledger. ATP-like energy is a separate pool.
    All constants are experimental model parameters, not measured fungal units.
    """

    def __init__(self, environment: Environment, config=None, *, seed=42, position=None):
        self.environment = environment
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.nodes = {}
        self.segments = {}
        self.tips = {}
        self.next_node = self.next_segment = self.next_tip = 0
        self.time = 0.0
        self.step_index = 0
        self.respired = self.carbon_added = self.carbon_removed = 0.0
        self.uptake = self.maintenance = self.growth_spent = 0.0
        self.energy_dissipated = 0.0
        self.memory_spent = 0.0
        self.memory_frozen = False
        self.pressure_residual = 0.0
        self.counts = Counter()
        self.events = deque(maxlen=self.config.event_limit)
        self.history = deque(maxlen=1000)
        h,w = environment.shape
        x,y = position if position is not None else ((w-1)/2,(h-1)/2)
        if not environment.inside(x,y) or environment.blocked[environment.cell(x,y)]:
            raise ValueError("inoculation position must be inside unblocked substrate")
        volume = math.pi*self.config.radius**2
        node = self._node(x,y,1.0,water=volume*1.2,nutrient=.6,energy=1.0,osmolyte=.25,reserve=.2,biomass=volume*self.config.growth_carbon)
        node.pressure = (node.water/node.volume-1)/self.config.wall_compliance
        for angle in (0, math.pi/2, math.pi, 3*math.pi/2):
            if len(self.tips)<self.config.max_tips:
                self._tip(node.id,math.cos(angle),math.sin(angle))
        self.initial_carbon = self.total_carbon()
        self.initial_water = self.total_water()
        self.initial_energy = sum(n.energy for n in self.nodes.values())
        self._validate()

    def _event(self, kind, **details):
        self.counts[kind] += 1
        self.events.append({"time":self.time,"type":kind,**details})

    def _node(self,x,y,length,**pools):
        n = Compartment(self.next_node,float(x),float(y),float(length),self.config.radius,**pools)
        self.nodes[n.id] = n
        self.next_node += 1
        return n

    def _tip(self,node,dx,dy):
        norm = math.hypot(dx,dy)
        if norm <= 0:
            raise ValueError("tip direction must be nonzero")
        tip = Tip(self.next_tip,node,dx/norm,dy/norm)
        self.tips[tip.id] = tip
        self.next_tip += 1
        return tip

    def connect(self,a,b,*,fusion=False):
        if a == b or a not in self.nodes or b not in self.nodes:
            raise ValueError("segment needs two existing distinct nodes")
        if not self.nodes[a].alive or not self.nodes[b].alive:
            raise ValueError("cannot connect dead compartments")
        pair = frozenset((a,b))
        for e in self.segments.values():
            if e.alive and frozenset((e.a,e.b)) == pair:
                return e,False
        na,nb = self.nodes[a],self.nodes[b]
        length = math.hypot(na.x-nb.x,na.y-nb.y)
        if length<1e-9:
            raise ValueError("cannot connect coincident compartments")
        e = Segment(self.next_segment,a,b,length,self.config.radius*.7,fusion=fusion)
        self.segments[e.id] = e
        self.next_segment += 1
        self._event("fusion" if fusion else "septation",segment=e.id,a=a,b=b)
        return e,True

    def total_carbon(self):
        living_and_dead = sum(n.nutrient+n.osmolyte+n.reserve+n.biomass+n.memory_material+n.fast_memory_material+n.polarity_material for n in self.nodes.values())
        return float(self.environment.nutrient.sum())+living_and_dead+sum(e.material+e.route_memory_material+e.route_fast_material for e in self.segments.values())+sum(t.pending_biomass for t in self.tips.values())+self.respired

    def total_water(self):
        return float(self.environment.water.sum())+sum(n.water for n in self.nodes.values())+sum(t.pending_water for t in self.tips.values())

    def _physiology(self,dt):
        c,env = self.config,self.environment
        temperature = math.exp(-((env.temperature-25)/18)**2)
        for n in self.nodes.values():
            if not n.alive:
                continue
            n.age += dt
            cell = env.cell(n.x,n.y)
            stress = float(env.stress[cell])
            oxygen = float(env.oxygen[cell])
            n.damage = min(1.0,max(0.0,n.damage+(stress*.08-n.energy*.02)*dt))
            local = env.sample(n.x,n.y)
            actual = env.consume(n.x,n.y,c.uptake_rate*n.volume*local/(.5+local)*temperature*(1-n.damage)*dt)
            n.nutrient += actual
            self.uptake += actual
            # Reserve exchange is an internal carbon transfer.
            target = .3*n.volume
            if n.nutrient>target:
                stored = min(n.nutrient-target,.08*n.volume*dt)
                n.nutrient -= stored
                n.reserve += stored
            else:
                released = min(n.reserve,.08*n.volume*dt,target-n.nutrient)
                n.reserve -= released
                n.nutrient += released
            fuel = min(n.nutrient,c.metabolic_rate*n.volume*temperature*oxygen*dt)
            n.nutrient -= fuel
            n.energy += fuel*c.energy_yield
            self.respired += fuel
            maintenance = min(n.energy,c.maintenance_rate*n.volume*(1+3*n.damage)*dt)
            n.energy -= maintenance
            self.maintenance += maintenance
            osmotic_target = .3*n.volume
            if n.osmolyte<osmotic_target:
                synthesized = min(n.nutrient,.025*n.volume*dt,osmotic_target-n.osmolyte)
                n.nutrient -= synthesized
                n.osmolyte += synthesized
            internal_pi = c.osmotic_scale*(n.nutrient+n.osmolyte)/max(n.water,1e-12)
            external_pi = c.osmotic_scale*env.concentration(n.x,n.y)
            wall_pressure = (n.water/n.volume-1)/c.wall_compliance
            # Substeps in step() limit osmotic overshoot. Reverse flux returns
            # water to the substrate rather than losing it to a hidden sink.
            request = c.membrane_rate*n.volume*(internal_pi-external_pi-wall_pressure)*(1-n.damage)*dt
            if request >= 0:
                n.water += env.consume(n.x,n.y,request,pool="water")
            else:
                returned = min(n.water,-request)
                n.water -= returned
                env.deposit(n.x,n.y,returned,pool="water")
            n.pressure = (n.water/n.volume-1)/c.wall_compliance
            self._polarity_memory(n,actual,dt)
        for e in self.segments.values():
            if e.alive:
                e.age += dt
                # One regulated septal pore per compartment connection.
                damage = max(self.nodes[e.a].damage,self.nodes[e.b].damage)
                target = 0.0 if damage>.6 else 1-damage
                e.pore_open += min(1.0,dt*2)*(target-e.pore_open)

    def _polarity_memory(self,n,uptake,dt):
        """Candidate uptake/polarity acclimation; not a measured fungal law.

        Only the local environmental gradient and actual uptake enter synthesis.
        No target angle or experiment label is available to this mechanism.
        """
        c=self.config
        if self.memory_frozen:
            return
        retained=math.exp(-c.foraging_memory_decay*dt)
        n.polarity_trace_x*=retained;n.polarity_trace_y*=retained
        recycled=n.polarity_material*(1-retained)
        n.polarity_material-=recycled;n.reserve+=recycled
        if c.foraging_memory_rate==0 or uptake<=0:
            return
        gx,gy=self.environment.gradient(n.x,n.y)
        magnitude=math.hypot(gx,gy)
        if magnitude<=1e-12:
            return
        reward=uptake/(.01*n.volume*dt+uptake)
        signal=magnitude/(.025+magnitude)
        rate=1-math.exp(-c.foraging_memory_rate*reward*signal*dt)
        dx=rate*(gx/magnitude-n.polarity_trace_x)
        dy=rate*(gy/magnitude-n.polarity_trace_y)
        requested=c.foraging_memory_cost*n.volume*math.hypot(dx,dy)
        energy_cost=.2
        material=min(requested,n.nutrient,n.energy/energy_cost)
        if material<=0:
            return
        fraction=material/requested
        n.polarity_trace_x+=fraction*dx;n.polarity_trace_y+=fraction*dy
        n.nutrient-=material;n.polarity_material+=material
        n.energy-=material*energy_cost;self.memory_spent+=material*energy_cost

    def _direction(self,tip,n,dt):
        gx,gy = self.environment.gradient(n.x,n.y)
        magnitude = math.hypot(gx,gy)
        dx,dy = tip.dx,tip.dy
        if magnitude>1e-12:
            weight = 1-math.exp(-dt*.18)
            dx = (1-weight)*dx+weight*gx/magnitude
            dy = (1-weight)*dy+weight*gy/magnitude
        strength=math.hypot(n.polarity_trace_x,n.polarity_trace_y)
        if strength>1e-12 and n.polarity_material>0 and self.config.foraging_memory_gain>0:
            weight=1-math.exp(-dt*self.config.foraging_memory_gain*strength)
            dx=(1-weight)*dx+weight*n.polarity_trace_x/strength
            dy=(1-weight)*dy+weight*n.polarity_trace_y/strength
        repel_x=repel_y=0.0
        for other in self.tips.values():
            if not other.active or other.id==tip.id or other.node==tip.node:
                continue
            target=self.nodes[other.node]
            rx,ry=n.x-target.x,n.y-target.y
            distance=math.hypot(rx,ry)
            if 1e-8<distance<1.5:
                repel_x+=rx/max(distance**2,.1)
                repel_y+=ry/max(distance**2,.1)
        repulsion=math.hypot(repel_x,repel_y)
        if repulsion>1e-12:
            dx+=.18*dt*repel_x/repulsion
            dy+=.18*dt*repel_y/repulsion
        norm = math.hypot(dx,dy)
        if norm>1e-12:
            dx,dy = dx/norm,dy/norm
        # Local trial directions provide obstacle avoidance without moving
        # through a blocked cell. No global route planner chooses the path.
        for angle in (0,.45,-.45,.9,-.9,1.5,-1.5):
            xx = dx*math.cos(angle)-dy*math.sin(angle)
            yy = dx*math.sin(angle)+dy*math.cos(angle)
            x,y = n.x+xx*.6,n.y+yy*.6
            if self.environment.path_clear(n.x,n.y,x,y):
                tip.dx,tip.dy = xx,yy
                return True
        return False

    def _growth(self,dt):
        c,env = self.config,self.environment
        for tip in list(self.tips.values()):
            if not tip.active:
                continue
            n = self.nodes[tip.node]
            tip.branch_age += dt
            # A lumped energy-dependent vesicle supply, explicitly a model
            # assumption rather than a molecular Spitzenkoerper simulation.
            tip.vesicles += (1-math.exp(-dt))*(n.energy/(.08*n.volume+n.energy)-tip.vesicles)
            rich = n.nutrient>.015*n.volume and n.energy>.01*n.volume
            tip.starvation = max(0.0,tip.starvation-dt) if rich else tip.starvation+dt
            if len(self.nodes)>=c.max_nodes or not self._direction(tip,n,dt):
                continue
            turgor = max(0.0,n.pressure)
            speed = c.growth_speed*turgor/(.2+turgor)*n.energy/(.06*n.volume+n.energy)*tip.vesicles*(1-n.damage)
            speed /= 1+float(env.resistance[env.cell(n.x,n.y)])
            distance = speed*dt
            area = math.pi*c.radius**2
            # Limit all costs against the same donor; growth never creates
            # nutrients, water, or ATP-like energy out of a sensory sample.
            volume = min(area*distance,n.nutrient/(c.growth_carbon+1e-12),n.energy/(c.growth_energy+1e-12),max(0,n.water-.65*n.volume)/1.08)
            distance = volume/area
            if distance < 1e-12:
                continue
            x,y = n.x+tip.dx*(tip.pending_length+distance),n.y+tip.dy*(tip.pending_length+distance)
            if not env.path_clear(n.x,n.y,x,y):
                continue
            carbon = c.growth_carbon*volume
            energy = c.growth_energy*volume
            n.nutrient -= carbon
            n.energy -= energy
            self.growth_spent += energy
            child_water = 1.08*volume
            n.water -= child_water
            tip.pending_length += distance
            tip.pending_water += child_water
            tip.pending_biomass += carbon
            n.pressure = (n.water/n.volume-1)/c.wall_compliance
            if tip.pending_length<c.compartment_length:
                continue
            distance = tip.pending_length
            volume = area*distance
            child_water = tip.pending_water
            fraction = min(.45,child_water/max(n.water,1e-12))
            pools = {}
            for name in ("nutrient","energy","osmolyte","reserve"):
                amount = getattr(n,name)*fraction
                setattr(n,name,getattr(n,name)-amount)
                pools[name] = amount
            child = self._node(x,y,distance,water=child_water,biomass=tip.pending_biomass,**pools)
            tip.pending_length=tip.pending_water=tip.pending_biomass=0.0
            # Memory polymer divides with the compartment; the directional
            # signal dilutes. This inheritance rule is an explicit hypothesis.
            inherited=n.polarity_material*fraction
            n.polarity_material-=inherited;child.polarity_material=inherited
            child.polarity_trace_x=n.polarity_trace_x*c.foraging_memory_inheritance
            child.polarity_trace_y=n.polarity_trace_y*c.foraging_memory_inheritance
            child.pressure = (child.water/child.volume-1)/c.wall_compliance
            self.connect(n.id,child.id)
            tip.node = child.id
            self._event("extension",tip=tip.id,node=child.id,length=distance)
        # Branch maturation survives extensions and uses its own clock.
        active_count = sum(t.active for t in self.tips.values())
        for tip in list(self.tips.values()):
            if not tip.active or active_count>=c.max_tips:
                continue
            n = self.nodes[tip.node]
            if tip.branch_age<c.branch_interval or n.nutrient<.04*n.volume or n.energy<.03*n.volume:
                continue
            signal = n.nutrient/(.15*n.volume+n.nutrient)
            if self.rng.random()<1-math.exp(-c.branch_rate*signal*dt):
                angle = self.rng.choice((-.7,.7))
                dx = tip.dx*math.cos(angle)-tip.dy*math.sin(angle)
                dy = tip.dx*math.sin(angle)+tip.dy*math.cos(angle)
                child = self._tip(n.id,dx,dy)
                tip.branch_age = 0.0
                active_count += 1
                self._event("branch",parent=tip.id,child=child.id,node=n.id)

    def _fusion(self):
        radius = self.config.fusion_radius
        if radius == 0:
            return
        grid = {}
        adjacency = {}
        for n in self.nodes.values():
            if n.alive:
                grid.setdefault((math.floor(n.x/radius),math.floor(n.y/radius)),[]).append(n.id)
        for e in self.segments.values():
            if e.alive:
                adjacency.setdefault(e.a,{})[e.b] = e.length
                adjacency.setdefault(e.b,{})[e.a] = e.length
        for tip in self.tips.values():
            if not tip.active or tip.branch_age<2:
                continue
            n = self.nodes[tip.node]
            excluded = set()
            distances = {n.id:0.0}
            frontier = [(0.0,n.id)]
            # Adjacent compartments within a short growing hypha are not a
            # new fusion opportunity merely because the discretization is fine.
            # Exclude a short arc-length neighborhood, independent of how
            # many small compartments happened to discretize that neighborhood.
            while frontier:
                distance,i = heapq.heappop(frontier)
                if i in excluded:
                    continue
                excluded.add(i)
                for j,length in adjacency.get(i,{}).items():
                    candidate = distance+length
                    if candidate<=radius*4 and candidate<distances.get(j,math.inf):
                        distances[j]=candidate
                        heapq.heappush(frontier,(candidate,j))
            cx,cy = math.floor(n.x/radius),math.floor(n.y/radius)
            candidates = []
            for gx in range(cx-1,cx+2):
                for gy in range(cy-1,cy+2):
                    for i in grid.get((gx,gy),()):
                        if i in excluded:
                            continue
                        target = self.nodes[i]
                        distance = math.hypot(n.x-target.x,n.y-target.y)
                        if 1e-8<distance<=radius and target.damage<.6 and n.damage<.6 and self.environment.path_clear(n.x,n.y,target.x,target.y):
                            candidates.append((distance,i))
            if candidates:
                _,i = min(candidates)
                e,created = self.connect(n.id,i,fusion=True)
                if created:
                    adjacency.setdefault(n.id,{})[i] = e.length
                    adjacency.setdefault(i,{})[n.id] = e.length
                    # Cytoplasmic continuity need not erase a growing front.
                    # A tip can continue extending after making a lateral link.

    def _remodel(self,dt):
        c,env = self.config,self.environment
        for tip in self.tips.values():
            if tip.active and tip.starvation>c.starvation_time:
                tip.active = False
                n=self.nodes[tip.node]
                env.deposit(n.x,n.y,tip.pending_biomass)
                env.deposit(n.x,n.y,tip.pending_water,pool="water")
                tip.pending_length=tip.pending_water=tip.pending_biomass=0.0
                self._event("tip_death",tip=tip.id,node=tip.node)
        for e in self.segments.values():
            if not e.alive:
                continue
            a,b = self.nodes[e.a],self.nodes[e.b]
            if e.activity>1e-5 and e.radius<c.radius:
                request = c.reinforcement_rate*e.activity*dt
                investment = min(request,a.nutrient*.05,b.nutrient*.05)
                a.nutrient -= investment/2
                b.nutrient -= investment/2
                e.material += investment
                e.radius = min(c.radius,e.radius+investment*.5)
            unused = e.activity<1e-5 and a.nutrient+b.nutrient<.008
            e.inactive_time = e.inactive_time+dt if unused else 0.0
        # Retract only genuinely dead terminal compartments. Interior bridge
        # removal requires a separate injury event; unused interior tissue is
        # not silently deleted or disconnected.
        degree = Counter()
        active_tip_nodes = {t.node for t in self.tips.values() if t.active}
        for e in self.segments.values():
            if e.alive:
                degree[e.a]+=1
                degree[e.b]+=1
        # Iterate over a stable snapshot because successful pruning releases
        # the terminal compartment and its conduit from bounded graph storage.
        for e in list(self.segments.values()):
            if not e.alive or e.inactive_time<c.starvation_time:
                continue
            endpoint = next((i for i in (e.a,e.b) if i != 0 and degree[i]==1 and i not in active_tip_nodes),None)
            if endpoint is None:
                continue
            n = self.nodes[endpoint]
            returned = n.nutrient+n.osmolyte+n.reserve+n.biomass+n.memory_material+n.fast_memory_material+n.polarity_material+e.material+e.route_memory_material+e.route_fast_material
            env.deposit(n.x,n.y,returned)
            env.deposit(n.x,n.y,n.water,pool="water")
            self.energy_dissipated += n.energy
            n.nutrient=n.osmolyte=n.reserve=n.biomass=n.memory_material=n.polarity_material=n.water=n.energy=0.0
            n.polarity_trace_x=n.polarity_trace_y=0.0
            n.receptor_trace=[]; n.receptor_variance=[]
            n.receptor_fast_trace=[];n.fast_memory_material=0.
            # Recycle storage only after all tissue carbon, water, and energy
            # have been accounted for. IDs remain monotonic, and the event log
            # preserves the removed IDs for auditability and saved-state replay.
            self.nodes.pop(endpoint)
            for segment_id, segment in list(self.segments.items()):
                if segment_id == e.id or (not segment.alive and endpoint in (segment.a,segment.b)):
                    self.segments.pop(segment_id)
            for tip_id, tip in list(self.tips.items()):
                if tip.node == endpoint and not tip.active:
                    self.tips.pop(tip_id)
            degree[e.a]-=1
            degree[e.b]-=1
            self._event("retraction",segment=e.id,node=endpoint)

    def _reactivate_tips(self):
        """Resume a dormant front when its living compartment is refueled.

        Tip state is treated as a dormant growth apparatus, not a new free
        inoculum. The node must be alive, metabolically viable, and locally
        exposed to enough substrate; subsequent extension still pays normal
        carbon, water, and energy costs. This is an explicit model hypothesis.
        """
        c, env = self.config, self.environment
        active_nodes = {t.node for t in self.tips.values() if t.active}
        active_count = sum(t.active for t in self.tips.values())
        if active_count >= c.max_tips:
            return
        dormant = {}
        for tip in self.tips.values():
            if not tip.active and tip.node not in dormant:
                dormant[tip.node] = tip
        candidates = []
        for node in self.nodes.values():
            if node.id in active_nodes or not node.alive:
                continue
            substrate = env.sample(node.x, node.y)
            if (substrate < c.resprout_threshold or
                    node.nutrient <= .015 * node.volume or
                    node.energy <= .01 * node.volume):
                continue
            # Favor viable, substrate-exposed compartments; stable ID breaks ties.
            score = substrate + node.nutrient/max(node.volume, 1e-12) + node.energy/max(node.volume, 1e-12)*1e-3
            candidates.append((-score, node.id, node))
        for _, node_id, node in sorted(candidates):
            if active_count >= c.max_tips:
                break
            tip = dormant.get(node_id)
            if tip is not None:
                tip.active = True
                tip.starvation = 0.0
                tip.branch_age = 0.0
                event = "tip_reactivation"
            else:
                gx, gy = env.gradient(node.x, node.y)
                norm = math.hypot(gx, gy)
                if norm <= 1e-12:
                    gx, gy = node.polarity_trace_x, node.polarity_trace_y
                    norm = math.hypot(gx, gy)
                if norm <= 1e-12:
                    angle = self.rng.random() * 2 * math.pi
                    gx, gy = math.cos(angle), math.sin(angle)
                tip = self._tip(node_id, gx, gy)
                event = "tip_resprout"
            active_nodes.add(node_id)
            active_count += 1
            self._event(event, tip=tip.id, node=node_id)

    def _advance(self,dt):
        self.environment.step(dt)
        self._physiology(dt)
        transport(self,dt)
        self._reactivate_tips()
        self._growth(dt)
        self._fusion()
        self._remodel(dt)
        self.time += dt

    def step(self,dt=None):
        dt = self.config.dt if dt is None else float(dt)
        if not math.isfinite(dt) or dt<=0:
            raise ValueError("dt must be finite and positive")
        self._validate()
        saved = deepcopy(self.__dict__)
        original_environment = self.environment
        try:
            # The hydraulic solver is implicit; membrane/growth updates use
            # bounded substeps. No giant dt can bypass accounting guards.
            count = max(1,math.ceil(dt/.5))
            for _ in range(count):
                self._advance(dt/count)
            self.step_index += 1
            self._validate()
            summary = self.summary()
            self.history.append(summary)
            return summary
        except Exception:
            original_environment.__dict__.clear()
            original_environment.__dict__.update(saved["environment"].__dict__)
            saved["environment"] = original_environment
            self.__dict__.clear()
            self.__dict__.update(saved)
            raise

    def run(self,steps):
        if isinstance(steps,bool) or not isinstance(steps,int) or steps<0:
            raise ValueError("steps must be a nonnegative integer")
        for _ in range(steps):
            self.step()
        return self.summary()

    def add_resource_patch(self,x,y,radius=6,strength=3):
        added = self.environment.add_patch(x,y,radius,strength)
        self.carbon_added += added
        self._event("resource_added",carbon=added)
        return added

    def deplete_region(self,x,y,radius):
        if radius<0 or not all(math.isfinite(v) for v in (x,y,radius)):
            raise ValueError("invalid depletion region")
        yy,xx = np.indices(self.environment.shape)
        mask = (xx-x)**2+(yy-y)**2<=radius**2
        removed = float(self.environment.nutrient[mask].sum())
        self.environment.nutrient[mask]=0
        self.carbon_removed += removed
        self._event("resource_removed",carbon=removed)
        return removed

    def _validate(self):
        env = self.environment
        if not isinstance(self.memory_frozen,bool):
            raise ValueError('invalid memory freeze flag')
        if env.blocked.shape!=env.shape or env.blocked.dtype!=np.dtype(bool):
            raise ValueError('invalid obstacle mask')
        for name in ('next_node','next_segment','next_tip','step_index'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError('invalid identity counter or clock')
        for name in ('time','respired','carbon_added','carbon_removed','uptake','maintenance','growth_spent','energy_dissipated','pressure_residual','initial_carbon','initial_water','initial_energy','memory_spent'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('invalid resource ledger or clock')
        for collection,counter in ((self.nodes,self.next_node),(self.segments,self.next_segment),(self.tips,self.next_tip)):
            if any(isinstance(i,bool) or not isinstance(i,int) or i<0 or obj.id!=i or i>=counter for i,obj in collection.items()):
                raise ValueError('invalid graph identity or allocation counter')
        for name in ("nutrient","water","oxygen","stress","resistance"):
            field = getattr(env,name)
            if field.shape!=env.shape or not np.all(np.isfinite(field)) or np.any(field<0):
                raise ValueError(f"invalid environmental {name}")
        if not math.isfinite(env.temperature):
            raise ValueError("invalid temperature")
        for n in self.nodes.values():
            values = (n.water,n.nutrient,n.energy,n.osmolyte,n.reserve,n.biomass,n.memory_material,n.fast_memory_material,n.polarity_material,n.damage,n.age)
            if not all(math.isfinite(v) and v>=-1e-10 for v in values) or n.volume<=0 or not math.isfinite(n.pressure):
                raise ArithmeticError(f"invalid compartment {n.id}")
            if not isinstance(n.alive,bool) or not all(math.isfinite(v) for v in (n.x,n.y,n.length,n.radius,n.volume)) or n.length<=0 or n.radius<=0 or n.damage>1:
                raise ArithmeticError('invalid compartment geometry or life state')
            if not all(math.isfinite(v) for v in (n.polarity_trace_x,n.polarity_trace_y)) or math.hypot(n.polarity_trace_x,n.polarity_trace_y)>1+1e-9:
                raise ArithmeticError("invalid polarity trace")
            if isinstance(n.memory_exposures,bool) or not isinstance(n.memory_exposures,int) or n.memory_exposures<0:
                raise ArithmeticError("invalid exposure count")
            if isinstance(n.fast_memory_exposures,bool) or not isinstance(n.fast_memory_exposures,int) or n.fast_memory_exposures<0 or len(n.receptor_fast_trace) not in (0,3):
                raise ArithmeticError('invalid fast memory dimensions or exposure count')
            if not all(math.isfinite(v) and 0<=v<=1 for v in n.receptor_fast_trace):
                raise ArithmeticError('invalid fast memory trace')
            if len(n.receptor_trace)!=len(n.receptor_variance) or len(n.receptor_trace) not in (0,3):
                raise ArithmeticError("invalid receptor dimensions")
            if not all(math.isfinite(v) and 0<=v<=1 for v in n.receptor_trace) or not all(math.isfinite(v) and v>=0 for v in n.receptor_variance):
                raise ArithmeticError("invalid receptor trace")
            if not env.inside(n.x,n.y):
                raise ArithmeticError(f"compartment outside substrate: {n.id}")
        pairs = set()
        for e in self.segments.values():
            if not isinstance(e.alive,bool) or not isinstance(e.fusion,bool) or any(isinstance(v,bool) or not isinstance(v,int) for v in (e.a,e.b)):
                raise ArithmeticError('invalid segment endpoints or life state')
            if not all(math.isfinite(v) and v>=0 for v in (e.age,e.inactive_time)):
                raise ArithmeticError('invalid segment age')
            if e.a not in self.nodes or e.b not in self.nodes or e.a==e.b or e.length<=0 or e.radius<=0:
                raise ArithmeticError(f"invalid segment {e.id}")
            if not math.isfinite(e.route_trace) or not 0<=e.route_trace<=1:
                raise ArithmeticError("invalid cord trace")
            if not math.isfinite(e.route_fast_trace) or not 0<=e.route_fast_trace<=1 or not all(math.isfinite(v) and v>=0 for v in (e.route_memory_material,e.route_fast_material)):
                raise ArithmeticError('invalid adaptive cord trace or material')
            if not all(math.isfinite(v) for v in (e.length,e.radius,e.flow,e.pore_open,e.activity,e.material)) or not 0<=e.pore_open<=1 or e.material<0:
                raise ArithmeticError(f"non-finite or invalid segment {e.id}")
            if e.alive:
                pair = frozenset((e.a,e.b))
                if pair in pairs or not self.nodes[e.a].alive or not self.nodes[e.b].alive:
                    raise ArithmeticError("duplicate edge or connection to dead tissue")
                pairs.add(pair)
        for t in self.tips.values():
            if not isinstance(t.active,bool) or isinstance(t.node,bool) or not isinstance(t.node,int):
                raise ArithmeticError('invalid tip identity or activity')
            if t.node not in self.nodes or (t.active and not self.nodes[t.node].alive):
                raise ArithmeticError(f"invalid tip {t.id}")
            if not all(math.isfinite(v) for v in (t.dx,t.dy,t.branch_age,t.starvation,t.vesicles,t.pending_length,t.pending_water,t.pending_biomass)) or min(t.pending_length,t.pending_water,t.pending_biomass)<0 or not math.isclose(math.hypot(t.dx,t.dy),1,abs_tol=1e-8):
                raise ArithmeticError(f"invalid tip direction or state {t.id}")
        if hasattr(self,"initial_carbon"):
            carbon_error = abs(self.total_carbon()-(self.initial_carbon+self.carbon_added-self.carbon_removed))
            water_error = abs(self.total_water()-self.initial_water)
            if carbon_error>1e-8*max(1,self.initial_carbon) or water_error>1e-8*max(1,self.initial_water):
                raise ArithmeticError(f"budget violation: carbon={carbon_error}, water={water_error}")
            expected_energy = self.initial_energy+self.respired*self.config.energy_yield-self.maintenance-self.growth_spent-self.memory_spent-self.energy_dissipated
            if abs(sum(n.energy for n in self.nodes.values())-expected_energy)>1e-8*max(1,abs(expected_energy)):
                raise ArithmeticError("ATP-like energy budget violation")

    def summary(self):
        nodes = [n for n in self.nodes.values() if n.alive]
        edges = [e for e in self.segments.values() if e.alive]
        return {"step":self.step_index,"time":self.time,"nodes":len(nodes),"stored_nodes":len(self.nodes),
                "segments":len(edges),"active_tips":sum(t.active for t in self.tips.values()),
                "total_length":sum(e.length for e in edges),"total_flow":sum(abs(e.flow) for e in edges),
                "mean_turgor":sum(max(0,n.pressure) for n in nodes)/max(1,len(nodes)),
                "mean_radius":sum(e.radius for e in edges)/max(1,len(edges)),
                "nutrient":sum(n.nutrient for n in nodes),"reserve":sum(n.reserve for n in nodes),
                "energy":sum(n.energy for n in nodes),"polarity_material":sum(n.polarity_material for n in nodes),
                "mean_polarity_memory":sum(math.hypot(n.polarity_trace_x,n.polarity_trace_y) for n in nodes)/max(1,len(nodes)),
                "memory_material":sum(n.memory_material+n.fast_memory_material for n in nodes)+sum(e.route_memory_material+e.route_fast_material for e in edges),"memory_spent":self.memory_spent,
                "biomass":sum(n.biomass for n in nodes)+sum(e.material for e in edges)+sum(t.pending_biomass for t in self.tips.values()),
                "uptake":self.uptake,"respired_carbon":self.respired,"events":dict(self.counts),
                "carbon_error":self.total_carbon()-(self.initial_carbon+self.carbon_added-self.carbon_removed),
                "water_error":self.total_water()-self.initial_water,"pressure_residual":self.pressure_residual,
                "energy_error":sum(n.energy for n in self.nodes.values())-(self.initial_energy+self.respired*self.config.energy_yield-self.maintenance-self.growth_spent-self.memory_spent-self.energy_dissipated)}

    def state_dict(self):
        env = self.environment
        return {"schema":1,"memory_frozen":self.memory_frozen,"config":self.config.as_dict(),"rng":self.rng.getstate(),
                "environment":{**{name:getattr(env,name).tolist() for name in ("nutrient","water","oxygen","stress","resistance","blocked")},"diffusion":env.diffusion,"temperature":env.temperature},
                "nodes":[asdict(n) for n in self.nodes.values()],"segments":[asdict(e) for e in self.segments.values()],"tips":[asdict(t) for t in self.tips.values()],
                "scalars":{name:getattr(self,name) for name in ("next_node","next_segment","next_tip","time","step_index","respired","carbon_added","carbon_removed","uptake","maintenance","growth_spent","energy_dissipated","pressure_residual","initial_carbon","initial_water","initial_energy","memory_spent")},
                "counts":dict(self.counts),"events":list(self.events),"history":list(self.history)}

    def save(self,path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(self.state_dict(),indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls,path):
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_state_dict(cls,data):
        if data.get("schema")!=1:
            raise ValueError("unsupported MYCELIA state schema")
        c = Config(**data["config"])
        e = data["environment"]
        env = Environment(e["nutrient"],e["water"],diffusion=e["diffusion"],temperature=e["temperature"])
        if np.asarray(e['blocked']).dtype!=np.dtype(bool):
            raise ValueError('invalid obstacle mask encoding')
        for name in ("oxygen","stress","resistance","blocked"):
            setattr(env,name,np.array(e[name],dtype=bool if name=="blocked" else float))
        self = cls.__new__(cls)
        self.memory_spent = 0.0
        self.memory_frozen = data.get("memory_frozen",False)
        self.environment,self.config = env,c
        def parse_records(records,record_type):
            parsed={}
            for record in records:
                identity=record['id']
                if isinstance(identity,bool) or not isinstance(identity,int) or identity<0 or identity in parsed:
                    raise ValueError('duplicate or invalid graph identity')
                parsed[identity]=record_type(**record)
            return parsed
        self.nodes = parse_records(data['nodes'],Compartment)
        self.segments = parse_records(data['segments'],Segment)
        self.tips = parse_records(data['tips'],Tip)
        allowed={'next_node','next_segment','next_tip','time','step_index','respired','carbon_added','carbon_removed','uptake','maintenance','growth_spent','energy_dissipated','pressure_residual','initial_carbon','initial_water','initial_energy','memory_spent'}
        if set(data['scalars'])-allowed or (allowed-{'memory_spent'})-set(data['scalars']):
            raise ValueError('invalid saved scalar fields')
        self.__dict__.update(data["scalars"])
        self.counts = Counter(data["counts"])
        self.events = deque(data["events"],maxlen=c.event_limit)
        self.history = deque(data["history"],maxlen=1000)
        def tuples(x):
            return tuple(tuples(i) for i in x) if isinstance(x,list) else x
        self.rng = random.Random()
        self.rng.setstate(tuples(data["rng"]))
        self._validate()
        return self
