"""Sandbox entry point: score one candidate config on the lab tiers."""
import json, sys
from .benchmark import LabBenchmark, scorecard

spec = json.loads(open(sys.argv[1]).read())
result = scorecard(spec['config'], LabBenchmark(spec['benchmark']), tuple(spec.get('seeds', (0, 1, 2))),
                   spec.get('robustness', True))
open(sys.argv[2], 'w').write(json.dumps(result))
