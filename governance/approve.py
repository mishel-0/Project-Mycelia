"""Human-only release actions. The lab never imports or calls this module.

  python -m governance.approve status
  python -m governance.approve bootstrap   --benchmark B --registry R
  python -m governance.approve approve ID  --benchmark B --registry R
  python -m governance.approve rollback    --benchmark B --registry R
"""
import argparse, getpass, json, sys
from .gate import Gate

DEFAULT_BASELINE = {'engine': {'gamma': 30.0, 'grow_margin': 0.5}, 'epochs': 2, 'mirror': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=('status', 'bootstrap', 'approve', 'rollback')); ap.add_argument('candidate', nargs='?')
    ap.add_argument('--benchmark', default='lab_state/benchmark'); ap.add_argument('--registry', default='lab_state/registry')
    ap.add_argument('--state', default='lab_state/governance')
    a = ap.parse_args(); gate = Gate(a.benchmark, a.registry, state_root=a.state)
    if a.action == 'status':
        print(json.dumps({'production': gate.production(), 'pending': [r for r in gate.releases.records()
                          if r.get('status') == 'awaiting_human']}, indent=2, default=str)); return
    if not sys.stdin.isatty():
        sys.exit('release actions need an interactive human terminal')
    who = getpass.getuser()
    target = a.candidate or a.action
    if input(f'Type "{a.action} {target}" to confirm: ').strip() != f'{a.action} {target}':
        sys.exit('not confirmed')
    if a.action == 'bootstrap':
        gate.bootstrap(DEFAULT_BASELINE, who)
    elif a.action == 'approve':
        gate.approve(a.candidate, who)
    else:
        gate.rollback(who)
    print('done:', json.dumps(gate.production(), indent=2))


if __name__ == '__main__':
    main()
