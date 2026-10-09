import hashlib, json
import numpy as np, pytest
from governance.gate import Gate, PolicyError
from mycelia.symbiosis_lab.benchmark import prepare
from mycelia.symbiosis_lab.registry import AppendLog

CFG = {'engine': {'gamma': 10.0}, 'epochs': 1, 'mirror': True}


def setup(tmp_path):
    rng = np.random.default_rng(0); c = rng.standard_normal((4, 6, 3))
    y = np.arange(200) % 4; x = (c[y] + .5 * rng.standard_normal((200, 6, 3))).astype(np.float32)
    prepare({'tr': x[:150], 'fl': x[:150], 'te': x[150:]}, y, np.arange(150), np.arange(150, 200), tmp_path / 'b')
    pol = tmp_path / 'policy.toml'; pol.write_bytes(open('governance/policy.toml', 'rb').read())
    ph = tmp_path / 'policy.sha256'; ph.write_text(hashlib.sha256(pol.read_bytes()).hexdigest())
    g = Gate(tmp_path / 'b', tmp_path / 'r', tmp_path / 's', pol, ph, tmp_path / 'STOP')
    g.bootstrap(CFG, 'tester'); return g


def verified(g):
    cid = g.registry.propose('baseline', CFG, 'h', 'b', 'm', {}, 'p'); g.registry.update(cid, 'verified'); return cid


def test_policy_tamper_is_refused(tmp_path):
    g = setup(tmp_path); g.policy_path.write_text(g.policy_path.read_text() + '\n# edited')
    with pytest.raises(PolicyError): g.policy()
    with pytest.raises(PolicyError): g.request_release(verified(g))


def test_release_needs_verification_and_uses_holdout_once(tmp_path):
    g = setup(tmp_path); cid = g.registry.propose('baseline', CFG, 'h', 'b', 'm', {}, 'p')
    with pytest.raises(PolicyError): g.request_release(cid)
    g.registry.update(cid, 'verified'); rec = g.request_release(cid)
    assert rec['status'] == 'awaiting_human' and g.production()['id'] == 'baseline'
    with pytest.raises(PolicyError): g.request_release(cid)


def test_only_approval_changes_production_and_rollback_restores(tmp_path):
    g = setup(tmp_path); cid = verified(g); g.request_release(cid)
    assert g.production()['id'] == 'baseline'
    g.approve(cid, 'human'); assert g.production()['id'] == cid
    g.rollback('human'); assert g.production()['id'] == 'baseline'
    with pytest.raises(PolicyError): g.approve('cand-unknown', 'human')


def test_stop_and_frozen_benchmark(tmp_path):
    g = setup(tmp_path); cid = verified(g); g.stop_file.write_text('stop')
    with pytest.raises(PolicyError): g.request_release(cid)
    g.stop_file.unlink(); h = tmp_path / 'b' / 'holdout.npz'; h.write_bytes(h.read_bytes() + b'x')
    with pytest.raises(PolicyError): g.request_release(cid)


def test_logs_are_tamper_evident(tmp_path):
    log = AppendLog(tmp_path / 'l.jsonl'); log.append({'a': 1}); log.append({'a': 2}); assert log.verify()
    lines = log.path.read_text().splitlines(); r = json.loads(lines[0]); r['a'] = 9
    log.path.write_text(json.dumps(r) + '\n' + lines[1] + '\n'); assert not log.verify()


def test_lab_package_never_imports_governance():
    import pathlib, re
    for f in pathlib.Path('mycelia/symbiosis_lab').glob('*.py'):
        assert not re.search(r'^\s*(from|import)\s+governance', f.read_text(), re.M), f
