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
    DC motor with pendulum
    """

    dataset_name = "dcmotor"
    params = argparse.Namespace(
        train_trials=1000,
        test_trials=10,
        train_steps=1000,
        test_steps=10000,
        initial_skip=0,
        dt=0.1,
        # number of elements
        n_k=1,  # spring           | capacitor
        n_m=2,  # mass             | inductor
        n_d=2,  # damper           | resistor
        n_g=0,  #                  | resistor
        n_s=1,  # external force   | voltage source
        n_b=0,  # moving boundary  | current source
        # domain 1
        n_k1=1,  # spring           | capacitor
        n_m1=1,  # mass             | inductor
        n_d1=1,  # damper           | resistor
        n_g1=0,  #                  | resistor
        n_s1=0,  # external force   | voltage source
        n_b1=0,  # moving boundary  | current source
        # domein 2
        n_k2=0,  # spring           | capacitor
        n_m2=1,  # mass             | inductor
        n_d2=1,  # damper           | resistor
        n_g2=0,  #                  | resistor
        n_s2=1,  # external force   | voltage source
        n_b2=0,  # moving boundary  | current source
        # number of observable states
        n_obs=3,
        # characteristics
        L=2.5,
        m=2.0,
        l=1.5,
        g=1.0,
        K=0.5,
        d1=lambda v: 0.02 * np.sign(v) * np.abs(v) ** (1 / 3),  # physical resistance
        R=lambda I: 0.05 * np.sign(I) * np.abs(I) ** (3),  # electrical resistance
        # initial state
        initial_dist_var_q=0.5,
        initial_dist_var_w=0.1,
        initial_dist_var_I=0.1,
        # external force
        fs_n_comp=5,
        fs_amp_max=0.3,
        fs_amp_min=0.1,
        fs_freq_max=0.3,
        fs_freq_min=0.05,
    )

    def __init__(self, base_dir="./", retry=False, **kwargs):
        data_train, data_test = self._get_dataset(base_dir, retry)
        self.data_train = data_train
        self.data_test = data_test
        self.data_mean = self.data_train.u.reshape(-1, self.params.n_obs).mean(axis=0)
        self.data_std = self.data_train.u.reshape(-1, self.params.n_obs).std(axis=0)
        f_s = self._external_effort(self.data_train.external_effort_param, self.data_train.t_eval, i=np.arange(self.data_train.u.shape[0]))
        self.v_b_mean = np.zeros([0])
        self.v_b_std = np.zeros([0])
        self.q_b_mean = np.zeros([0])
        self.q_b_std = np.zeros([0])
        self.f_s_mean = f_s.reshape(-1).mean(keepdims=True)
        self.f_s_std = f_s.reshape(-1).std(keepdims=True)

    def external_flow(self, t, i, train=True, pos=False):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def external_position(self, *args, **kwargs):
        return self.external_flow(*args, pos=True, **kwargs)

    def external_effort(self, t, i, train=True):
        data = self.data_train if train else self.data_test
        if not hasattr(data, "external_effort_param_gpu"):
            data.external_effort_param_gpu = tuple([torch.tensor(p, device=t.device, dtype=t.dtype) for p in data.external_effort_param])
        force_freq, force_amplitude, force_phase = data.external_effort_param_gpu
        t = t.unsqueeze(-1)
        omega = 2 * torch.pi * force_freq[i]
        val = (force_amplitude[i] * torch.sin(omega * t + force_phase[i])).sum(-1).unsqueeze(-1)
        return val

    def _external_effort(self, params, t, i=None):
        force_freq, force_amplitude, force_phase = params
        non_array_t = isinstance(t, (float, int))
        if i is None:
            i = slice(None)
        if non_array_t:
            t = np.array([t])
        t = t[:, None, None]
        omega = 2 * np.pi * force_freq[i]
        val = (force_amplitude[i] * np.sin(omega * t + force_phase[i])).sum(-1)
        if non_array_t:
            val = val[0]
        return val

    def visualize(self, base_dir, state_predicted):
        os.makedirs(f"{base_dir}") if not os.path.exists(f"{base_dir}") else None
        # time series, time, state
        state_predicted = state_predicted.transpose(1, 0, 2)
        print(f"Generating images of predicted results.")
        for idx_solution in range(self.data_test.u.shape[0]):
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=100)
            q, w, I = self.data_test.u[idx_solution].T
            qp, wp, Ip = state_predicted[idx_solution].T
            f_s = self._external_effort(self.data_test.external_effort_param, self.data_test.t_eval, i=idx_solution).flatten()
            t_eval = self.data_test.t_eval
            ax.plot(t_eval, q, color="C0", ls="-", alpha=0.3)
            ax.plot(t_eval, w, color="C1", ls="-", alpha=0.3)
            ax.plot(t_eval, I, color="C2", ls="-", alpha=0.3)
            ax.plot(t_eval, qp, color="C0", ls="-", label="q")
            ax.plot(t_eval, wp, color="C1", ls="-", label="w")
            ax.plot(t_eval, Ip, color="C2", ls="-", label="I")
            ax.plot(t_eval, f_s, color="C3", ls="-", label="f_s")
            ax.legend(loc="right")
            plt.savefig(f"{base_dir}/predicted_{idx_solution}.png", dpi=100)
            ax.set_xlim([t_eval[0], t_eval[-1] / 10])
            plt.savefig(f"{base_dir}/short_predicted_{idx_solution}.png", dpi=300)
            plt.close()

    def _get_dataset(self, base_dir, retry, **kwargs):
        path = f"{base_dir}/{self.dataset_name}-dataset.pkl"
        try:
            if retry:
                assert False
            data_train, data_test = pickle.load(open(path, "rb"))
            print("Successfully loaded data from {}".format(path))
            return data_train, data_test
        except:
            np.random.seed(111)
            print("Failed to load data from {}. Regenerating dataset...".format(path))

            print("Generating training data.")
            data_train = self._make_orbits(trials=self.params.train_trials, n_steps=self.params.train_steps, **kwargs)
            data_train.phase = "train"
            print("Generating test data.")
            data_test = self._make_orbits(trials=self.params.test_trials, n_steps=self.params.test_steps, **kwargs)
            data_test.phase = "test"

            pickle.dump((data_train, data_test), open(path, "wb"))
            os.makedirs(f"{base_dir}/{self.dataset_name}") if not os.path.exists(f"{base_dir}/{self.dataset_name}") else None
            for data in (data_train, data_test):
                phase = data.phase
                print(f"Generating images of {phase} data.")
                for itr in range(data.u.shape[0]):
                    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=100)
                    q, w, I = data.u[itr].T
                    f_s = self._external_effort(data.external_effort_param, data.t_eval, i=itr).flatten()
                    t_eval = data.t_eval
                    ax.plot(t_eval, q, color="C0", ls="-", label="q")
                    ax.plot(t_eval, w, color="C1", ls="-", label="w")
                    ax.plot(t_eval, I, color="C2", ls="-", label="I")
                    ax.plot(t_eval, f_s, color="C3", ls="-", label="f_s")
                    ax.legend()
                    plt.savefig(f"{base_dir}/{self.dataset_name}/{phase}_{itr}.png", dpi=100)
                    plt.close()

        return data_train, data_test

    def _get_initial_states(self, trials):
        q = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_q  # angle of pendulam
        w = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_w  # anglar velocity
        I = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_I  # current
        return np.stack([q, w, I], axis=1)

    def _get_external_effort_params(self, trials):
        force_freq = np.random.uniform(self.params.fs_freq_min, self.params.fs_freq_max, size=(trials, self.params.fs_n_comp))
        force_amplitude = np.random.uniform(self.params.fs_amp_min, self.params.fs_amp_max, size=(trials, self.params.fs_n_comp))
        force_phase = np.random.uniform(0, 2 * np.pi, size=(trials, self.params.fs_n_comp))
        return force_freq, force_amplitude, force_phase

    def _ode(self, t, state, external_effort_params):
        q, w, I = state.T
        L, m, l, g, K, D1, R = self.params.L, self.params.m, self.params.l, self.params.g, self.params.K, self.params.d1, self.params.R
        f_s = self._external_effort(external_effort_params, t)  # external voltage
        T_D = D1(w)  # torque
        V_D = R(I)
        qdot = w
        wdot = (-m * g * l * np.sin(q) + K * I - T_D) / (m * l**2)
        Idot = (-w * K - V_D + f_s) / L
        return np.stack([qdot, wdot, Idot], axis=1)

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
