"""Resource-paid, local chemical imprints in real compartment/cord state.

The receptor encoding and class-conditioned rewards are computational hypotheses,
not evidence that fungi see MRI images. Recognition is direct graph matching; no
SVM, CNN, backpropagation, or separately fitted readout is used here.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from copy import deepcopy
import json,math
from pathlib import Path
import numpy as np
from .environment import Environment
from .organism import Mycelium
from .state import Config


def image_cues(gray):
    """Fixed grayscale-to-chemical-cue interface; no learned image encoder."""
    gray=np.asarray(gray,dtype=float)
    if gray.ndim!=2 or min(gray.shape)<3 or not np.all(np.isfinite(gray)) or np.any((gray<0)|(gray>1)):
        raise ValueError('gray must be a finite 2D image in [0,1]')
    gy,gx=np.gradient(gray)
    return np.stack((gray,.5+.5*gx,.5+.5*gy),axis=-1)


@dataclass(frozen=True)
class MemoryConfig:
    size:int=16
    colonies_per_label:int=8
    learning_rate:float=.12
    material_cost:float=.002
    energy_cost:float=.2
    decay_rate:float=.003
    cord_rate:float=.002
    physiology_interval:int=20
    consolidation:bool=False
    adaptive:bool=False
    fast_rate:float=.45
    slow_rate:float=.015
    stability_scale:float=.02
    fast_decay_rate:float=.03
    route_memory_cost:float=.002
    def __post_init__(self):
        if not isinstance(self.consolidation,bool) or not isinstance(self.adaptive,bool):raise ValueError('memory modes must be boolean')
        for name in ('size','colonies_per_label','physiology_interval'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError(f'invalid {name}')
        if self.size<3:raise ValueError('size must be >=3')
        for name in ('learning_rate','material_cost','energy_cost','decay_rate','cord_rate','fast_rate','slow_rate','stability_scale','fast_decay_rate','route_memory_cost'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError(f'invalid {name}')
        if not 0<self.learning_rate<=1 or self.material_cost<=0 or self.energy_cost<=0:
            raise ValueError('learning rate and synthesis costs must be positive')
        if max(self.fast_rate,self.slow_rate)>1 or self.stability_scale<=0 or self.route_memory_cost<=0:
            raise ValueError('invalid adaptive rates or costs')
        if self.adaptive and self.fast_rate+self.slow_rate<=0:raise ValueError('adaptive mode needs a trace')


class MemoryColony:
    """A pre-inoculated lattice of actual Mycelium compartments and septal cords.

    The fixed scaffold makes image sensing reproducible. Normal free-tip growth
    remains in Mycelium; the learning experiment does not pretend the scaffold
    grew from one inoculation point. Trace synthesis consumes real nutrient and
    ATP-like energy; cord reinforcement changes real radius and conductance.
    """
    def __init__(self,config=None):
        self.config=config or MemoryConfig();s=self.config.size
        self.organism=Mycelium(Environment(np.zeros((s,s))),Config(max_nodes=s*s,max_tips=1),position=(0,0))
        o=self.organism;o.tips.clear();volume=o.nodes[0].volume
        for y in range(s):
            for x in range(s):
                if x or y:o._node(x,y,1,water=volume*1.2,nutrient=.6,energy=1,osmolyte=.25,reserve=.2,biomass=volume*o.config.growth_carbon)
        for y in range(s):
            for x in range(s):
                i=y*s+x
                if x+1<s:o.connect(i,i+1)
                if y+1<s:o.connect(i,i+s)
        o.initial_carbon=o.total_carbon();o.initial_water=o.total_water();o.initial_energy=sum(n.energy for n in o.nodes.values());o._validate()
        self.exposures=0

    def _cue(self,cue):
        cue=np.asarray(cue,dtype=float)
        if cue.shape!=(self.config.size,self.config.size,3) or not np.all(np.isfinite(cue)) or np.any((cue<0)|(cue>1)):
            raise ValueError('cue must match scaffold size, with three finite channels in [0,1]')
        return cue.reshape(-1,3)

    def expose(self,cue,*,reward=1.,feed=.02):
        # Exposure is transactional, including trace synthesis before physiology.
        original=self.organism;env=original.environment;before=deepcopy(original.__dict__);count=self.exposures
        try:return self._expose(cue,reward=reward,feed=feed)
        except Exception:
            saved_env=before.pop("environment")
            env.__dict__.clear();env.__dict__.update(saved_env.__dict__)
            original.__dict__.clear();original.__dict__.update(before);original.environment=env
            self.exposures=count
            raise

    def _expose(self,cue,*,reward=1.,feed=.02):
        o=self.organism;c=self.config;x=self._cue(cue)
        if o.memory_frozen:raise RuntimeError('memory is frozen')
        if not math.isfinite(reward) or not 0<=reward<=1 or not math.isfinite(feed) or feed<0:raise ValueError('invalid reward or feed')
        if reward==0:return {'changed_nodes':0,'synthesis_carbon':0.,'synthesis_energy':0.}
        # A reward is an externally supplied nutrient cue, not a natural class label.
        # Each actual addition is recorded, never assigned to a hidden pool.
        living=[n for n in o.nodes.values() if n.alive]
        if not living:return {'changed_nodes':0,'synthesis_carbon':0.,'synthesis_energy':0.}
        added=feed*len(living)
        for n in living:o.environment.nutrient[o.environment.cell(n.x,n.y)]+=feed
        o.carbon_added+=added
        changed=0;carbon=energy=0.
        for i,n in o.nodes.items():
            if not n.alive:continue
            if c.adaptive:
                intake=o.environment.consume(n.x,n.y,feed)
                n.nutrient+=intake;o.uptake+=intake
                updates,spent=self._adaptive_node(n,x[i],reward)
                changed+=updates;carbon+=spent;energy+=spent*c.energy_cost
                continue
            old=np.asarray(n.receptor_trace if n.receptor_trace else [.5,.5,.5])
            delta=x[i]-old
            base_rate=(min(c.learning_rate,1./(n.memory_exposures+1)) if c.consolidation else c.learning_rate)
            rate=(1. if not n.memory_exposures else base_rate)*reward
            request=c.material_cost*rate*(.05+float(np.abs(delta).mean()))
            # Local uptake plus stored resources pays for receptor/transcript turnover.
            intake=o.environment.consume(n.x,n.y,feed)
            n.nutrient+=intake;o.uptake+=intake
            affordable=min(request,n.nutrient,n.energy/c.energy_cost)
            if request>0 and affordable<=0:continue
            fraction=1. if request==0 else affordable/request
            rate*=fraction
            new=old+rate*delta
            variance=np.asarray(n.receptor_variance if n.receptor_variance else [.04]*3)
            variance=(1-rate)*variance+rate*delta**2
            n.nutrient-=affordable;n.memory_material+=affordable
            spent=affordable*c.energy_cost;n.energy-=spent;o.memory_spent+=spent
            n.receptor_trace=new.tolist();n.receptor_variance=variance.tolist();n.memory_exposures+=1
            changed+=int(np.any(new!=old));carbon+=affordable;energy+=spent
        for e in o.segments.values():
            if not e.alive:continue
            a,b=o.nodes[e.a],o.nodes[e.b]
            if not (a.receptor_trace or (c.adaptive and a.receptor_fast_trace)) or not (b.receptor_trace or (c.adaptive and b.receptor_fast_trace)):continue
            contrast=float(np.abs(x[e.a]-x[e.b]).mean())
            # Only local co-exposure drives consolidation; no error backpropagation.
            similarity=math.exp(-4*contrast)
            request=c.cord_rate*similarity*reward
            investment=min(request,2*a.nutrient,2*b.nutrient,2*a.energy/c.energy_cost,2*b.energy/c.energy_cost,max(0.,(o.config.radius-e.radius)*2))
            if investment>0:
                a.nutrient-=investment/2;b.nutrient-=investment/2;e.material+=investment
                spent=investment*c.energy_cost
                a.energy-=spent/2;b.energy-=spent/2;o.memory_spent+=spent
                carbon+=investment;energy+=spent
                e.radius=min(o.config.radius,e.radius+investment*.5)
                if not c.adaptive:e.route_trace+=(1-math.exp(-c.learning_rate*reward))*(contrast-e.route_trace)
            if c.adaptive:
                spent=self._adaptive_cord(e,contrast,reward)
                carbon+=spent;energy+=spent*c.energy_cost
        self.exposures+=1
        if self.exposures%c.physiology_interval==0:o.step()
        o._validate()
        return {'changed_nodes':changed,'synthesis_carbon':carbon,'synthesis_energy':energy}

    def _adaptive_node(self,n,cue,reward):
        """Fast response plus slower acquisition gated by local cue stability.

        These are uncalibrated hypotheses. Every channel update is separately
        funded; no exposure counter can permanently shut off later plasticity.
        """
        c=self.config;o=self.organism
        previous_fast=np.asarray(n.receptor_fast_trace if n.receptor_fast_trace else [.5]*3)
        stability=math.exp(-float(np.mean((cue-previous_fast)**2))/c.stability_scale) if n.receptor_fast_trace and c.fast_rate>0 else 1.
        changed=False;total=0.
        for trace,material,count,base in (
            ('receptor_fast_trace','fast_memory_material','fast_memory_exposures',c.fast_rate),
            ('receptor_trace','memory_material','memory_exposures',c.slow_rate)):
            if base==0:continue
            values=getattr(n,trace);old=np.asarray(values if values else [.5]*3)
            initialized=bool(values) and getattr(n,material)>0 and getattr(n,count)>0
            rate=(base*(stability if trace=='receptor_trace' else 1.) if initialized else 1.)*reward
            request=c.material_cost*rate*(.05+float(np.mean(np.abs(cue-old))))
            paid=min(request,n.nutrient,n.energy/c.energy_cost)
            if paid<=0:continue
            actual=rate*paid/request;new=old+actual*(cue-old)
            n.nutrient-=paid;setattr(n,material,getattr(n,material)+paid)
            n.energy-=paid*c.energy_cost;o.memory_spent+=paid*c.energy_cost
            setattr(n,trace,new.tolist());setattr(n,count,getattr(n,count)+1)
            if trace=='receptor_trace':
                variance=np.asarray(n.receptor_variance if n.receptor_variance else [.04]*3)
                n.receptor_variance=((1-actual)*variance+actual*(cue-old)**2).tolist()
            changed|=bool(np.any(new!=old));total+=paid
        return int(changed),total

    def _adaptive_cord(self,e,contrast,reward):
        """Paid chemical turnover is independent of available radius growth."""
        c=self.config;o=self.organism;a,b=o.nodes[e.a],o.nodes[e.b]
        stability=math.exp(-(contrast-e.route_fast_trace)**2/c.stability_scale) if e.route_fast_material>0 and c.fast_rate>0 else 1.
        total=0.
        for trace,material,base in (('route_fast_trace','route_fast_material',c.fast_rate),('route_trace','route_memory_material',c.slow_rate)):
            if base==0:continue
            old=getattr(e,trace)
            rate=(base*(stability if trace=='route_trace' else 1.) if getattr(e,material)>0 else 1.)*reward
            request=c.route_memory_cost*rate*(.05+abs(contrast-old))
            paid=min(request,2*a.nutrient,2*b.nutrient,2*a.energy/c.energy_cost,2*b.energy/c.energy_cost)
            if paid<=0:continue
            a.nutrient-=paid/2;b.nutrient-=paid/2
            a.energy-=paid*c.energy_cost/2;b.energy-=paid*c.energy_cost/2;o.memory_spent+=paid*c.energy_cost
            setattr(e,material,getattr(e,material)+paid)
            setattr(e,trace,old+rate*(paid/request)*(contrast-old));total+=paid
        return total

    def has_trace(self,kind='slow'):
        if kind=='fast':
            return self.config.adaptive and any(n.alive and n.fast_memory_exposures and n.fast_memory_material>0 and n.receptor_fast_trace for n in self.organism.nodes.values())
        return any(n.alive and n.memory_exposures and n.memory_material>0 and n.receptor_trace for n in self.organism.nodes.values())

    def distance(self,cue):
        """Read-only chemical mismatch + conductance-weighted cord mismatch."""
        x=self._cue(cue)
        return min((self._distance_channel(x,kind) for kind in ('slow','fast') if self.has_trace(kind)),default=math.inf)

    def _distance_channel(self,x,kind):
        o=self.organism;field='receptor_fast_trace' if kind=='fast' else 'receptor_trace'
        traces=np.array([getattr(o.nodes[i],field) or [.5]*3 for i in range(self.config.size**2)])
        alive=np.array([o.nodes[i].alive for i in range(self.config.size**2)])
        node_error=float(np.mean((x[alive]-traces[alive])**2))
        edges=[e for e in o.segments.values() if e.alive]
        aa=np.array([e.a for e in edges],dtype=int);bb=np.array([e.b for e in edges],dtype=int)
        weights=np.array([e.conductance(o.config.hydraulic_scale) for e in edges])
        contrasts=np.abs(x[aa]-x[bb]).mean(axis=1) if edges else np.array([])
        errors=(contrasts-np.array([e.route_fast_trace if kind=='fast' else e.route_trace for e in edges]))**2
        route_error=float(np.dot(weights,errors)/max(float(weights.sum()),1e-12))
        return node_error+.15*route_error

    def rest(self,duration):
        """Transactional chemical turnover and conservative material recycling."""
        original=self.organism;env=original.environment;before=deepcopy(original.__dict__)
        try:return self._rest(duration)
        except Exception:
            saved_env=before.pop('environment')
            env.__dict__.clear();env.__dict__.update(saved_env.__dict__)
            original.__dict__.clear();original.__dict__.update(before);original.environment=env
            raise

    def _rest(self,duration):
        o=self.organism
        if o.memory_frozen:raise RuntimeError('memory is frozen')
        if not math.isfinite(duration) or duration<0:raise ValueError('invalid rest duration')
        fraction=1-math.exp(-self.config.decay_rate*duration)
        fast_fraction=1-math.exp(-self.config.fast_decay_rate*duration)
        for n in o.nodes.values():
            if n.receptor_trace:n.receptor_trace=(.5+(np.asarray(n.receptor_trace)-.5)*(1-fraction)).tolist()
            released=n.memory_material*fraction;n.memory_material-=released;n.reserve+=released
            if n.receptor_fast_trace:n.receptor_fast_trace=(.5+(np.asarray(n.receptor_fast_trace)-.5)*(1-fast_fraction)).tolist()
            released=n.fast_memory_material*fast_fraction;n.fast_memory_material-=released;n.reserve+=released
        for e in o.segments.values():
            recycled=e.material*fraction;e.material-=recycled
            o.nodes[e.a].reserve+=recycled/2;o.nodes[e.b].reserve+=recycled/2
            e.radius=max(o.config.radius*.7,e.radius-recycled*.5)
            e.route_trace*=1-fraction
            for field,trace,decay in (('route_memory_material','route_trace',fraction),('route_fast_material','route_fast_trace',fast_fraction)):
                released=getattr(e,field)*decay;setattr(e,field,getattr(e,field)-released)
                o.nodes[e.a].reserve+=released/2;o.nodes[e.b].reserve+=released/2
                if trace=='route_fast_trace':e.route_fast_trace*=1-decay
        o._validate()

    def freeze(self):self.organism.memory_frozen=True
    def state_dict(self):return {'config':asdict(self.config),'exposures':self.exposures,'organism':self.organism.state_dict()}
    @classmethod
    def from_state(cls,data):
        # Mycelium parses state without temporary disk files.
        self=cls.__new__(cls);self.config=MemoryConfig(**data['config']);self.exposures=data['exposures']
        if isinstance(self.exposures,bool) or not isinstance(self.exposures,int) or self.exposures<0:raise ValueError('invalid colony exposure count')
        self.organism=Mycelium.from_state_dict(data['organism'])
        size=self.config.size
        if set(self.organism.nodes)!=set(range(size*size)) or self.organism.environment.shape!=(size,size):raise ValueError('invalid memory scaffold')
        if any(n.x!=i%size or n.y!=i//size or max(n.memory_exposures,n.fast_memory_exposures)>self.exposures for i,n in self.organism.nodes.items()):raise ValueError('invalid memory cue map or exposure count')
        return self


class AssociativeMycelium:
    """Rewarded colony populations: online local imprints, direct graph recall.

    Labels select the rewarded population during teaching. Recognition uses the
    smallest local mismatch within each population. This is an explicit prototype
    model implemented inside biological graph state, not a claim about fungal MRI
    recognition and not an independently trained classifier.
    """
    def __init__(self,labels,config=None):
        self.config=config or MemoryConfig();self.labels=tuple(labels)
        if not self.labels or any(not isinstance(v,str) or not v.strip() for v in self.labels) or len(set(self.labels))!=len(self.labels):raise ValueError('labels must be unique nonempty strings')
        self.colonies={label:[] for label in self.labels};self.frozen=False
    def learn(self,cue,label):
        if self.frozen:raise RuntimeError('memory is frozen')
        if label not in self.colonies:raise ValueError('unknown teaching label')
        cue=np.asarray(cue,dtype=float)
        if cue.shape!=(self.config.size,self.config.size,3) or not np.all(np.isfinite(cue)) or np.any((cue<0)|(cue>1)):raise ValueError('invalid cue')
        population=self.colonies[label]
        if len(population)<self.config.colonies_per_label:
            colony=MemoryColony(self.config)
            result=colony.expose(cue)
            # Registration is committed only after successful, paid exposure.
            if result['synthesis_carbon']>0:population.append(colony)
            return result
        colony=min(population,key=lambda z:z.distance(cue))
        return colony.expose(cue)
    def scores(self,cue):
        self._validate_cues(np.asarray(cue,dtype=float)[None,...])
        # Canonical label order also defines ties in batch recall. JSON mapping
        # order or a reordered population dictionary must not change a decision.
        return {label:min((z.distance(cue) for z in self.colonies[label]),default=math.inf) for label in self.labels}
    def predict(self,cue):
        scores=self.scores(cue)
        if all(math.isinf(v) for v in scores.values()):return None
        return min(scores,key=scores.get)
    def _validate_cues(self,cues):
        s=self.config.size
        if cues.ndim!=4 or cues.shape[1:]!=(s,s,3) or not np.all(np.isfinite(cues)) or np.any((cues<0)|(cues>1)):
            raise ValueError('cue batch must be finite N x size x size x 3 in [0,1]')

    def score_many(self,cues,*,batch_size=256):
        """Same graph mismatch as scores(), vectorized; no fitted readout/cache.

        Arrays are rebuilt from actual compartment/cord state on every call.
        Direct ablation or graph edits therefore cannot leave a stale predictor.
        """
        cues=np.asarray(cues,dtype=float);self._validate_cues(cues)
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:raise ValueError('invalid batch size')
        result=np.full((len(cues),len(self.labels)),np.inf)
        colonies=[]
        for label_index,label in enumerate(self.labels):
            for colony in self.colonies[label]:
                for kind in ('slow','fast'):
                    if colony.has_trace(kind):colonies.append((label_index,colony,kind))
        if not colonies:return result
        pairs=sorted({(e.a,e.b) for _,colony,_ in colonies for e in colony.organism.segments.values() if e.alive})
        pair_index={pair:i for i,pair in enumerate(pairs)}
        aa=np.array([a for a,b in pairs],dtype=int);bb=np.array([b for a,b in pairs],dtype=int)
        traces=[];node_weights=[];weights=np.zeros((len(colonies),len(pairs)));routes=np.zeros_like(weights)
        for j,(_,colony,kind) in enumerate(colonies):
            organism=colony.organism
            field='receptor_fast_trace' if kind=='fast' else 'receptor_trace'
            traces.append([getattr(organism.nodes[i],field) or [.5]*3 for i in range(self.config.size**2)])
            living=np.array([organism.nodes[i].alive for i in range(self.config.size**2)],dtype=float)
            node_weights.append(np.repeat(living,3)/(3*living.sum()))
            for e in organism.segments.values():
                if e.alive:
                    k=pair_index[(e.a,e.b)]
                    weights[j,k]=e.conductance(organism.config.hydraulic_scale);routes[j,k]=e.route_fast_trace if kind=='fast' else e.route_trace
        traces=np.asarray(traces).reshape(len(colonies),-1)
        node_weights=np.asarray(node_weights)
        weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
        trace_norm=np.sum(node_weights*traces**2,axis=1)
        route_norm=np.sum(weights*routes**2,axis=1)
        for start in range(0,len(cues),batch_size):
            flat=cues[start:start+batch_size].reshape(-1,self.config.size**2,3)
            x=flat.reshape(len(flat),-1)
            node_error=np.maximum(0,(x**2)@node_weights.T+trace_norm[None,:]-2*x@(node_weights*traces).T)
            contrast=np.abs(flat[:,aa]-flat[:,bb]).mean(axis=2)
            route_error=np.maximum(0,contrast**2@weights.T-2*contrast@(weights*routes).T+route_norm[None,:])
            distances=node_error+.15*route_error
            for j,(label_index,_,_) in enumerate(colonies):
                result[start:start+len(flat),label_index]=np.minimum(result[start:start+len(flat),label_index],distances[:,j])
        return result

    def predict_many(self,cues,*,batch_size=256):
        scores=self.score_many(cues,batch_size=batch_size)
        return [None if not np.any(np.isfinite(row)) else self.labels[int(np.argmin(row))] for row in scores]

    def freeze(self):
        self.frozen=True
        for population in self.colonies.values():
            for colony in population:colony.freeze()
    def state_dict(self):return {'schema':'mycelia-associative-1','labels':self.labels,'config':asdict(self.config),'frozen':self.frozen,'colonies':{k:[z.state_dict() for z in v] for k,v in self.colonies.items()}}
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(self.state_dict(),allow_nan=False));tmp.replace(path)
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        if data.get('schema')!='mycelia-associative-1':raise ValueError('unsupported memory schema')
        if not isinstance(data.get('frozen'),bool):raise ValueError('invalid frozen flag')
        self=cls(data['labels'],MemoryConfig(**data['config']));self.frozen=data['frozen']
        if set(data['colonies'])!=set(self.labels):raise ValueError('invalid label population')
        self.colonies={k:[MemoryColony.from_state(z) for z in v] for k,v in data['colonies'].items()}
        if any(len(v)>self.config.colonies_per_label for v in self.colonies.values()):raise ValueError('too many colonies')
        if any(colony.config!=self.config for v in self.colonies.values() for colony in v):raise ValueError('inconsistent colony configuration')
        if self.frozen and any(not colony.organism.memory_frozen for v in self.colonies.values() for colony in v):raise ValueError('inconsistent frozen population')
        if self.frozen:self.freeze()
        return self
