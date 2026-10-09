import numpy as np
from mycelia.receptor_memory import ReceptorConfig, ReceptorField, TraceMemory


def images(seed, n=40):
    rng = np.random.default_rng(seed); out = []; labels = []
    for i in range(n):
        img = rng.random((16, 16)) * .2; c = i % 2
        if c: img[4:12, 4:12] += .7
        else: img[:, ::4] += .7
        out.append(np.clip(img, 0, 1).astype(np.float32)); labels.append(c)
    return out, np.array(labels)


def test_unsupervised_field_and_teach_only_memory_generalize():
    x, y = images(0); xt, yt = images(1)
    field = ReceptorField(ReceptorConfig(receptors=16, patch=4, patches_per_image=10, iterations=5)).fit(x)
    f = field.encode_many(x)
    assert f.shape[0] == len(x) and np.all(np.isfinite(f))
    mem = TraceMemory(2).fit(f, y, [(field.encode_many([i[:, ::-1].copy() for i in x]), y)])
    assert (mem.predict(field.encode_many(xt)) == yt).mean() >= .9


def test_field_is_deterministic():
    x, _ = images(0, 10); c = ReceptorConfig(receptors=8, patch=4, patches_per_image=5, iterations=3)
    assert np.array_equal(ReceptorField(c).fit(x).receptors, ReceptorField(c).fit(x).receptors)
