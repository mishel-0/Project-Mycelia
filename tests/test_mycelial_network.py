import numpy as np
from mycelia.receptor_memory import ReceptorConfig, ReceptorField
from mycelia.mycelial_network import MycelialNetwork, NetworkConfig


def setup(**kw):
    rng = np.random.default_rng(0); imgs = [rng.random((32, 32)).astype(np.float32) for _ in range(12)]
    for i in imgs: i[:, :16] += i[:, 16:].mean()  # correlated regions
    imgs = [np.clip(i, 0, 1) for i in imgs]
    f = ReceptorField(ReceptorConfig(receptors=8, patch=4, patches_per_image=10, iterations=3)).fit(imgs)
    net = MycelialNetwork(f, NetworkConfig(size=4, **kw))
    return net, np.stack([net.sense(i) for i in imgs])


def test_cord_learning_is_paid_and_budgets_balance():
    net, s = setup(); before = net.organism.memory_spent
    out = net.learn_cords(s); summary = net.summary()
    assert out['cords_reinforced'] > 0 and net.organism.memory_spent > before
    assert summary['cord_radius_max'] > summary['cord_radius_min']
    assert abs(summary['carbon_error']) < 1e-8 and abs(summary['energy_error']) < 1e-8


def test_spread_follows_topology():
    net, s = setup(); x = np.zeros_like(s[:1]); x[0, 0] = 1  # signal in corner compartment
    settled = net.settle(x).reshape(1, 16, -1)
    assert settled[0, 1].sum() > 0 and settled[0, 15].sum() == 0  # neighbour gets signal, far corner not
    isolated, _ = setup(steps=0)
    assert np.array_equal(isolated.settle(x), x.reshape(1, -1))
