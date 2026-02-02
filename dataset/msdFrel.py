import numpy as np
from dataset.msdFabs import Dataset as DatasetAbs

class Dataset(DatasetAbs):
    """
    Mass-spring-damper system with external force in relative coordinate system
    """

    dataset_name = "msdFrel"

    def __init__(self, base_dir="./", retry=False, **kwargs):
        super(Dataset, self).__init__(base_dir, retry, **kwargs)

    def _make_orbits(self, trials, n_steps, atol=1e-7, rtol=1e-9, **kwargs):
        data = super(Dataset, self)._make_orbits(trials, n_steps, atol, rtol, **kwargs)
        u = data.u
        q_k, v_m = np.split(u, 2, axis=-1)
        q_k_rel = np.stack([q_k[..., i] - q_k[..., i - 1] if i > 0 else q_k[..., i] for i in range(self.params.n_k)], axis=-1)
        u = np.concatenate([q_k_rel, v_m], axis=-1)

        data.u = u

        return data
