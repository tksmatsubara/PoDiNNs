import numpy as np
import pickle
import torch
import os
import scipy.integrate
import matplotlib as mpl
import argparse

mpl.use("Agg")
import matplotlib.pyplot as plt

class Dataset:
    """
    Mass-spring-damper system with external force in absolute coordinate system
    """

    dataset_name = "msdFabs"

    params = argparse.Namespace(
        train_trials=1000,
        test_trials=10,
        train_steps=1000,
        test_steps=10000,
        initial_skip=0,
        dt=0.1,
        # number of elements
        n_k=3,  # spring           | capacitor
        n_m=3,  # mass             | inductor
        n_d=2,  # damper           | resistor
        n_g=0,  #                  | resistor
        n_s=1,  # external force   | voltage source
        n_b=0,  # moving boundary  | current source
        # number of observable states
        n_obs=3 * 2,
        # characteristics
        m=[1.0 + 0.2 * i for i in range(3)],
        k=[
            lambda q: 0.4 * q + 0.1 * q**3,
            lambda q: 0.3 * q + 0.1 * q**3,
            lambda q: 0.2 * q + 0.1 * q**3,
        ],
        d=[
            lambda v: -0.01 * np.sign(v) * np.abs(v) ** (1 / 3),
            lambda v: v * 0,
            lambda v: -0.05 * np.sign(v) * np.abs(v) ** (1 / 3),
        ],
        # initial state
        initial_range_q_k=0.5,
        initial_range_v_m=0.3,
        # external force
        fs_n_comp=3,
        fs_freq_max=0.2,
        fs_freq_min=0.1,
        fs_amp_max=0.4,
        fs_amp_min=0.2,
    )

    def __init__(self, base_dir="./", retry=False, **kwargs):
        data_train, data_test = self._get_dataset(base_dir, retry)
        self.data_train = data_train
        self.data_test = data_test
        self.data_mean = self.data_train.u.reshape(-1, self.data_train.u.shape[-1]).mean(axis=0)
        self.data_std = self.data_train.u.reshape(-1, self.data_train.u.shape[-1]).std(axis=0)
        f_s = self._external_effort(self.data_train.external_effort_param, self.data_train.t_eval, i=np.arange(self.data_train.u.shape[0]))
        self.v_b_mean = np.zeros([0])
        self.v_b_std = np.zeros([1])
        self.q_b_mean = np.zeros([0])
        self.q_b_std = np.zeros([1])
        self.f_s_mean = f_s.reshape(-1).mean(keepdims=True)
        self.f_s_std = f_s.reshape(-1).std(keepdims=True)

    def external_flow(self, t, i, train=True):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def external_position(self, t, i, train=True):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def external_effort(self, t, i, train=True):
        data = self.data_train if train else self.data_test
        if not hasattr(data, "external_effort_param_gpu"):
            data.external_effort_param_gpu = tuple([torch.tensor(p, device=t.device, dtype=t.dtype) for p in data.external_effort_param])
        ground_freq, ground_amplitude, ground_phase = data.external_effort_param_gpu
        t = t.unsqueeze(-1)
        omega = 2 * torch.pi * ground_freq[i]
        val = (ground_amplitude[i] * torch.cos(omega * t + ground_phase[i])).sum(-1).unsqueeze(-1)
        return val

    def _external_effort(self, params, t, i=None, pos=False):
        ground_freq, ground_amplitude, ground_phase = params
        non_array_t = isinstance(t, (float, int))
        if i is None:
            i = slice(None)
        if non_array_t:
            t = np.array([t])
        t = t[:, None, None]
        omega = 2 * np.pi * ground_freq[i]
        val = (ground_amplitude[i] * np.cos(omega * t + ground_phase[i])).sum(-1)
        if non_array_t:
            val = val[0]
        return val

    def visualize(self, base_dir, state_predicted):
        os.makedirs(f"{base_dir}") if not os.path.exists(f"{base_dir}") else None
        # time series, time, state
        state_predicted = state_predicted.transpose(1, 0, 2)
        print(f"Generating images of predicted results.")
        for idx_solution in range(self.data_test.u.shape[0]):
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=300)
            states = self.data_test.u[idx_solution].T
            predicted = state_predicted[idx_solution].T
            t_eval = self.data_test.t_eval
            f_s = self._external_effort(self.data_test.external_effort_param, self.data_test.t_eval, i=idx_solution).flatten()
            ax.plot(t_eval, f_s, color="k", ls="-", label="f_s")
            for i, s in enumerate(states):
                ax.plot(t_eval, s, color=f"C{i}", ls="-", alpha=0.3)
            for i, s in enumerate(predicted):
                if i < len(predicted) // 2:
                    ax.plot(t_eval, s, color=f"C{i}", ls="-", label=f"q_k{i}")
                else:
                    ax.plot(t_eval, s, color=f"C{i}", ls="-", label=f"v_m{i-len(predicted)//2}")
            ax.legend(loc="right")
            plt.savefig(f"{base_dir}/predicted_{idx_solution}.png", dpi=300)
            ax.set_xlim([t_eval[0], t_eval[-1] / 10])
            plt.savefig(f"{base_dir}/short_predicted_{idx_solution}.png", dpi=300)
            plt.close()

    def _get_dataset(self, base_dir, retry=False, **kwargs):
        path = f"{base_dir}/{self.dataset_name}-dataset.pkl"
        try:
            if retry:
                assert False
            data_train, data_test = pickle.load(open(path, "rb"))
            print("Successfully loaded data from {}".format(path))
            return data_train, data_test
        except:
            np.random.seed(99)
            print("Failed to load data from {}. Regenerating dataset...".format(path))
            print("Generating training data.")
            data_train = self._make_orbits(trials=self.params.train_trials, n_steps=self.params.train_steps, **kwargs)
            data_train.phase = "train"
            print("Generating test data.")
            data_test = self._make_orbits(trials=self.params.test_trials, n_steps=self.params.test_steps, **kwargs)
            data_test.phase = "test"
            pickle.dump((data_train, data_test), open(path, "wb"))
            os.makedirs(f"{base_dir}/{self.dataset_name}") if not os.path.exists(f"{base_dir}/{self.dataset_name}") else None
            for data in (data_test, data_train):
                phase = data.phase
                print(f"Generating images of {phase} data.")
                for itr in range(data.u.shape[0]):
                    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=300)
                    states = data.u[itr].T
                    f_s = self._external_effort(data.external_effort_param, data.t_eval, i=itr).flatten()
                    t_eval = data.t_eval
                    ax.plot(t_eval, f_s, color="k", ls="-", label="f_s")
                    for i, s in enumerate(states):
                        if i < len(states) // 2:
                            ax.plot(t_eval, s, color=f"C{i}", ls="-", label=f"q_k{i}")
                        else:
                            ax.plot(t_eval, s, color=f"C{i}", ls="-", label=f"v_m{i-len(states)//2}")
                    ax.legend()
                    plt.savefig(f"{base_dir}/{self.dataset_name}/{phase}_{itr}.png", dpi=300)
                    plt.close()

        return data_train, data_test

    def _get_initial_states(self, trials):
        assert self.params.n_k == self.params.n_m
        n = self.params.n_k
        q_k = np.random.uniform(-1, 1, [n, trials]) * self.params.initial_range_q_k  # displacement of spring
        v_m = np.random.uniform(-1, 1, [n, trials]) * self.params.initial_range_v_m  # velocity of mass
        return np.stack([*q_k, *v_m], axis=1)

    def _get_external_effort_params(self, trials):
        ground_freq = np.random.uniform(self.params.fs_freq_min, self.params.fs_freq_max, size=(trials, self.params.fs_n_comp))
        ground_amplitude = np.random.uniform(self.params.fs_amp_min, self.params.fs_amp_max, size=(trials, self.params.fs_n_comp))
        ground_phase = np.random.uniform(0, 2 * np.pi, size=(trials, self.params.fs_n_comp))
        return ground_freq, ground_amplitude, ground_phase

    def _ode(self, t, state, external_effort_param):
        assert self.params.n_k == self.params.n_m
        n = self.params.n_k
        q_k = state.T[:n]
        v_m = state.T[n:]
        m, k, d = self.params.m, self.params.k, self.params.d
        f_s = self._external_effort(external_effort_param, t)  # volocities, moving ground
        q_dot = v_m
        v_d = [v_m[i] - v_m[i - 1] if i > 0 else v_m[i] for i in range(n)]
        q_k_rel = [q_k[i] - q_k[i - 1] if i > 0 else q_k[i] for i in range(n)]
        effort_k = [k[i](q_k_rel[i]) for i in range(n)]
        effort_d = [d[i](v_d[i]) for i in range(n)]

        f = [None for i in range(n)]
        for i in range(n):
            f[i] = -effort_k[i] + effort_d[i]
        for i in range(n - 1):
            f[i] += +effort_k[i + 1] - effort_d[i + 1]
        f[-1] += f_s

        v_dot = [f[i] / m[i] for i in range(n)]
        return np.stack([*q_dot, *v_dot], axis=1)

    def _make_orbits(self, trials, n_steps, atol=1e-7, rtol=1e-9, **kwargs):
        initial_state = self._get_initial_states(trials)
        external_effort_param = self._get_external_effort_params(trials)

        t_eval = np.arange(-self.params.initial_skip, n_steps + 1) * self.params.dt
        t_span = [t_eval[0], t_eval[-1]]

        orbits = scipy.integrate.solve_ivp(fun=lambda t, state: self._ode(t, state.reshape(initial_state.shape), external_effort_param).flatten(), t_span=t_span, y0=initial_state.flatten(), t_eval=t_eval, atol=atol, rtol=rtol, **kwargs)
        assert orbits.success

        u = orbits.y.reshape(initial_state.shape[0], initial_state.shape[1], self.params.initial_skip + n_steps + 1).transpose([0, 2, 1])

        u = u[:, self.params.initial_skip :]
        t_eval = t_eval[self.params.initial_skip :]

        data = argparse.Namespace(
            u=u,
            t_eval=t_eval,
            n_trials=u.shape[0],
            n_steps=u.shape[1],
            initial_state=initial_state,
            external_effort_param=external_effort_param,
        )

        return data
