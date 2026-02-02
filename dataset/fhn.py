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
    FitzhHugh-Nagmo Model
    """

    dataset_name = "fhn"
    params = argparse.Namespace(
        train_trials=30,
        test_trials=10,
        train_steps=1000,
        test_steps=10000,
        initial_skip=0,
        dt=0.1,
        # number of elements
        n_k=1,  # spring           | capacitor
        n_m=1,  # mass             | inductor
        n_d=1,  # damper           | resistor
        n_g=1,  #                  | resistor
        n_s=0,  # external force   | voltage source
        n_b=1,  # moving boundary  | current source
        # number of observable states
        n_obs=2,
        # characteristics
        L=1 / 0.08,
        R=0.8,
        C=1.0,
        E=-0.7,
        D=lambda v: v**3 / 3 - v,
        dDdV=lambda v: v**2 - 1,
        # initial state
        initial_range_V=3.0,
        initial_range_W=3.0,
        # external flow
        vb_amp_max=1.5,
        vb_amp_min=0.1,
    )

    def __init__(self, base_dir="./", retry=False, **kwargs):
        data_train, data_test = self._get_dataset(base_dir, retry)
        self.data_train = data_train
        self.data_test = data_test
        self.data_mean = self.data_train.u.reshape(-1, self.params.n_obs).mean(axis=0)
        self.data_std = self.data_train.u.reshape(-1, self.params.n_obs).std(axis=0)
        v_b = self._external_flow(self.data_train.external_flow_param, self.data_train.t_eval, i=None)
        self.v_b_mean = v_b.reshape(-1).mean(keepdims=True)
        self.v_b_std = v_b.reshape(-1).std(keepdims=True)
        self.q_b_mean = np.zeros([0])
        self.q_b_std = np.zeros([1])
        self.f_s_mean = np.zeros([0])
        self.f_s_std = np.zeros([1])

    def external_effort(self, t, i, train=True):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def external_position(self, t, i, train=True):
        return torch.zeros(*t.shape, 1, device=t.device, dtype=t.dtype)

    def external_flow(self, t, i, train=True):
        data = self.data_train if train else self.data_test
        if not hasattr(data, "external_flow_param_gpu"):
            data.external_flow_param_gpu = torch.tensor(data.external_flow_param, device=t.device, dtype=t.dtype)
        ground_amplitude = data.external_flow_param_gpu
        val = ground_amplitude[i].unsqueeze(-1)
        return val

    def _external_flow(self, params, t, i=None):
        ground_amplitude = params
        if i is None:
            if isinstance(t, (float, int)):
                return ground_amplitude
            else:
                return ground_amplitude.repeat(t.shape)
        if isinstance(i, int) and isinstance(t, np.ndarray):
            return ground_amplitude[i].repeat(t.shape)
        raise NotImplementedError

    def visualize(self, base_dir, state_predicted):
        os.makedirs(f"{base_dir}") if not os.path.exists(f"{base_dir}") else None
        # time series, time, state
        state_predicted = state_predicted.transpose(1, 0, 2)
        print(f"Generating images of predicted results.")
        for idx_solution in range(self.data_test.u.shape[0]):
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=100)
            V, W = self.data_test.u[idx_solution].T
            Vp, Wp = state_predicted[idx_solution].T
            I = self._external_flow(self.data_test.external_flow_param, self.data_test.t_eval, i=idx_solution).flatten()
            t_eval = self.data_test.t_eval
            ax.plot(t_eval, I, color="k", ls="-", label="I")
            ax.plot(t_eval, V, color="C0", ls="-", alpha=0.3)
            ax.plot(t_eval, W, color="C1", ls="-", alpha=0.3)
            ax.plot(t_eval, Vp, color="C0", ls="-", label="V")
            ax.plot(t_eval, Wp, color="C1", ls="-", label="W")
            ax.legend(loc="right")
            plt.savefig(f"{base_dir}/predicted_{idx_solution}.png", dpi=100)
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
                    fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=100)
                    V, W = data.u[itr].T
                    I = self._external_flow(data.external_flow_param, data.t_eval, i=itr).flatten()
                    t_eval = data.t_eval
                    ax.plot(t_eval, I, color="k", ls="-", label="I")
                    ax.plot(t_eval, V, color="C0", ls="-", label="V")
                    ax.plot(t_eval, W, color="C1", ls="-", label="W")
                    ax.legend()
                    plt.savefig(f"{base_dir}/{self.dataset_name}/{phase}_{itr}.png", dpi=100)
                    plt.close()

        return data_train, data_test

    def _get_initial_states(self, trials):
        V = np.random.uniform(-1, 1, trials) * self.params.initial_range_V
        W = np.random.uniform(-1, 1, trials) * self.params.initial_range_W
        return np.stack([V, W], axis=1)

    def _get_external_flow_params(self, trials):
        ground_amplitude = np.linspace(
            self.params.vb_amp_min,
            self.params.vb_amp_max,
            trials + 2,
        )[1:-1]
        return ground_amplitude

    def _ode(self, t, state, external_flow_param):
        V, W = state.T
        L, R, C, E, D = self.params.L, self.params.R, self.params.C, self.params.E, self.params.D
        I = self._external_flow(external_flow_param, t)  # volocities, moving ground
        dV = (-D(V) - W + I) / C
        dW = (V - E - R * W) / L
        return np.stack([dV, dW], axis=1)

    def _make_orbits(self, trials, n_steps, atol=1e-7, rtol=1e-9, **kwargs):
        initial_state = self._get_initial_states(trials)
        external_flow_param = self._get_external_flow_params(trials)

        t_eval = np.arange(-self.params.initial_skip, n_steps + 1) * self.params.dt
        t_span = [t_eval[0], t_eval[-1]]

        orbits = scipy.integrate.solve_ivp(fun=lambda t, state: self._ode(t, state.reshape(initial_state.shape), external_flow_param).flatten(), t_span=t_span, y0=initial_state.flatten(), t_eval=t_eval, atol=atol, rtol=rtol, **kwargs)
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
            external_flow_param=external_flow_param,
        )

        return data
