import argparse
from pathlib import Path
from .engine import Hypha
from .report import write_run_report
from .experiments import run_validation


def main(argv=None):
    parser = argparse.ArgumentParser(description='MYCELIA mechanistic hypha (dimensional, uncalibrated)')
    parser.add_argument('--duration', type=float, default=30, help='additional simulated minutes')
    parser.add_argument('--dt', type=float, default=.1, help='public step duration in minutes')
    parser.add_argument('--resume', type=Path, help='mechanistic state.json, not a prototype state')
    parser.add_argument('--output', type=Path, default=Path('runs/biology'))
    parser.add_argument('--validate', action='store_true', help='run the fixed matched intervention panel')
    args = parser.parse_args(argv)
    try:
        if args.validate:
            if args.resume or args.duration != 30 or args.dt != .1:
                parser.error('--validate uses its fixed protocol; omit --resume, --duration and --dt overrides')
            result = run_validation(args.output)
            print(f'{sum(result["checks"].values())}/{len(result["checks"])} computational checks passed. Empirical validation: not performed.')
            print(args.output / 'report.html')
            return 0 if result['passed'] else 2
        h = Hypha.load(args.resume) if args.resume else Hypha()
        h.run(args.duration, args.dt)
        write_run_report(args.output, h)
        print(f'{h.time_min:.3f} min; {len(h.cells)} compartments; permanent extension {h.summary()["permanent_extension_um"]:.6f} um')
        print(args.output / 'report.html')
        return 0
    except (ValueError, ArithmeticError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'MYCELIA biology: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
