from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import scenario
from .organism import Mycelium
from .report import write_report
from .state import Config


def parser():
    p = argparse.ArgumentParser(description='MYCELIA: coupled compartment physiology and network growth')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--dt',type=float,default=None,help='Model time per step; default 0.5')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--scenario',choices=('patches','uniform','poor','obstacle'),default='patches')
    p.add_argument('--max-nodes',type=int,default=None)
    p.add_argument('--config',type=Path,help='JSON object containing Config overrides')
    p.add_argument('--resume',type=Path,help='Continue a saved state.json')
    p.add_argument('--output',type=Path,default=Path('runs/demo'))
    p.add_argument('--print-every',type=int,default=50)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.steps<0 or args.print_every<=0:
            raise ValueError('steps must be nonnegative; print-every must be positive')
        if args.resume:
            if args.config or args.max_nodes is not None:
                raise ValueError('resume uses its saved configuration; config/max-nodes cannot override it')
            organism = Mycelium.load(args.resume)
        else:
            options = json.loads(args.config.read_text()) if args.config else {}
            if not isinstance(options,dict):
                raise ValueError('config must be a JSON object')
            if args.dt is not None:
                options['dt'] = args.dt
            if args.max_nodes is not None:
                options['max_nodes'] = args.max_nodes
            config = Config(**options)
            organism = Mycelium(scenario(args.scenario,args.size,config.diffusion),config,seed=args.seed)
        print('MYCELIA 1.0 — compartment physiology → network growth')
        for i in range(args.steps):
            s = organism.step(args.dt)
            if (i+1)%args.print_every==0 or i+1==args.steps:
                print(f"step={s['step']:4d} compartments={s['nodes']:4d} tips={s['active_tips']:3d} branches={s['events'].get('branch',0):3d} fusions={s['events'].get('fusion',0):3d} flow={s['total_flow']:.5f} carbon_error={s['carbon_error']:.2e}")
        report = write_report(organism,args.output)
        print(f'Report: {report.resolve()}')
        print(f'State:  {(args.output/"state.json").resolve()}')
        return 0
    except (ValueError,TypeError,ArithmeticError,OSError,json.JSONDecodeError) as exc:
        p.exit(1,f'MYCELIA failed: {exc}\n')

