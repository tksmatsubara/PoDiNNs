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
    2D mass-spring system in relative coordinate
    """

    dataset_name = "ms2Dim"
    params = argparse.Namespace(
        train_trials=1000,
        test_trials=10,
        train_steps=1000,
        test_steps=10000,
        initial_skip=0,
        dt=0.1,
        # number of elements
        n_k=10, # spring           | capacitor
        n_m=4,  # mass             | inductor
        n_d=0,  # damper           | resistor
        n_g=0,  #                  | resistor
        n_s=0,  # external force   | voltage source
        n_b=0,  # moving boundary  | current source
        # number of observable states
        n_obs=14,
        # characteristics
        m1=5.0,
        m2=3.0,
        l1=3.0,
        l3=4.0,
        l4=5.0,
        k1=lambda q: 2.5 * q + 3.4 * q**3,
        k2=lambda q: 3.0 * q + 0.5 * q**3,
        k3=lambda q: 2.1 * q + 4.1 * q**3,
        k4=lambda q: 3.5 * q + 2.4 * q**3,
        k5=lambda q: 2.5 * q + 1.6 * q**3,
        # initial state
        initial_dist_var_qx_k1=0.5,
        initial_dist_var_qy_k1=0.5,
        initial_dist_var_qx_k2=0.5,
        initial_dist_var_qy_k2=0.5,
        initial_dist_var_vx_m1=0.1,
        initial_dist_var_vy_m1=0.1,
        initial_dist_var_vx_m2=0.1,
        initial_dist_var_vy_m2=0.1,
    )

    def __init__(self, base_dir="./", retry=False, **kwargs):
        data_train, data_test = self._get_dataset(base_dir, retry)
        self.data_train = data_train
        self.data_test = data_test
        self.data_mean = self.data_train.u.reshape(-1, self.params.n_obs).mean(axis=0)
        self.data_std = self.data_train.u.reshape(-1, self.params.n_obs).std(axis=0)
        self.v_b_mean = np.zeros([0])
        self.v_b_std = np.zeros([0])
        self.q_b_mean = np.zeros([0])
        self.q_b_std = np.zeros([0])
        self.f_s_mean = np.zeros([0])
        self.f_s_std = np.zeros([0])

    def external_flow(self, t, i, train=True):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def external_position(self, t, i, train=True):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def external_effort(self, t, i, train=True):
        return torch.zeros(*t.shape, 0, device=t.device, dtype=t.dtype)

    def visualize(self, base_dir, state_predicted):
        os.makedirs(f"{base_dir}") if not os.path.exists(f"{base_dir}") else None
        # time series, time, state
        state_predicted = state_predicted.transpose(1, 0, 2)
        print(f"Generating images of predicted results.")
        for idx_solution in range(self.data_test.u.shape[0]):
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(6, 3), facecolor=None, dpi=100)
            qx_k1, qy_k1, qx_k2, qy_k2, qx_k3, qy_k3, qx_k4, qy_k4, qx_k5, qy_k5, vx_m1, vy_m1, vx_m2, vy_m2 = self.data_test.u[idx_solution].T
            qxpk1, qypk1, qxpk2, qypk2, qxpk3, qypk3, qxpk4, qypk4, qxpk5, qypk5, vxpm1, vypm1, vxpm2, vypm2 = state_predicted[idx_solution].T
            t_eval = self.data_test.t_eval
            ax.plot(t_eval, qx_k1, color="C0", ls="-", alpha=0.3)
            ax.plot(t_eval, qy_k1, color="C1", ls="-", alpha=0.3)
            ax.plot(t_eval, qx_k2, color="C2", ls="-", alpha=0.3)
            ax.plot(t_eval, qy_k2, color="C3", ls="-", alpha=0.3)
            ax.plot(t_eval, qx_k3, color="C4", ls="-", alpha=0.3)
            ax.plot(t_eval, qy_k3, color="C5", ls="-", alpha=0.3)
            ax.plot(t_eval, qx_k4, color="C6", ls="-", alpha=0.3)
            ax.plot(t_eval, qy_k4, color="C7", ls="-", alpha=0.3)
            ax.plot(t_eval, qx_k5, color="C8", ls="-", alpha=0.3)
            ax.plot(t_eval, qy_k5, color="C9", ls="-", alpha=0.3)
            ax.plot(t_eval, vx_m1, color="C10", ls="-", alpha=0.3)
            ax.plot(t_eval, vy_m1, color="C11", ls="-", alpha=0.3)
            ax.plot(t_eval, vx_m2, color="C12", ls="-", alpha=0.3)
            ax.plot(t_eval, vy_m2, color="C13", ls="-", alpha=0.3)
            ax.plot(t_eval, qxpk1, color="C0", ls="-", label="qx_k1")
            ax.plot(t_eval, qypk1, color="C1", ls="-", label="qy_k1")
            ax.plot(t_eval, qxpk2, color="C2", ls="-", label="qx_k2")
            ax.plot(t_eval, qypk2, color="C3", ls="-", label="qy_k2")
            ax.plot(t_eval, qxpk3, color="C4", ls="-", label="qx_k3")
            ax.plot(t_eval, qypk3, color="C5", ls="-", label="qy_k3")
            ax.plot(t_eval, qxpk4, color="C6", ls="-", label="qx_k4")
            ax.plot(t_eval, qypk4, color="C7", ls="-", label="qy_k4")
            ax.plot(t_eval, qxpk5, color="C8", ls="-", label="qx_k5")
            ax.plot(t_eval, qypk5, color="C9", ls="-", label="qy_k5")
            ax.plot(t_eval, vxpm1, color="C10", ls="-", label="vx_m1")
            ax.plot(t_eval, vypm1, color="C11", ls="-", label="vy_m1")
            ax.plot(t_eval, vxpm2, color="C12", ls="-", label="vx_m2")
            ax.plot(t_eval, vypm2, color="C13", ls="-", label="vy_m2")
            ax.legend(loc="right")
            plt.savefig(f"{base_dir}/predicted_{idx_solution}.png", dpi=100)
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
                    qx_k1, qy_k1, qx_k2, qy_k2, qx_k3, qy_k3, qx_k4, qy_k4, qx_k5, qy_k5, vx_m1, vy_m1, vx_m2, vy_m2 = data.u[itr].T
                    t_eval = data.t_eval
                    ax.plot(t_eval, qx_k1, color="C0", ls="-", label="qx_k1")
                    ax.plot(t_eval, qy_k1, color="C1", ls="-", label="qy_k1")
                    ax.plot(t_eval, qx_k2, color="C2", ls="-", label="qx_k2")
                    ax.plot(t_eval, qy_k2, color="C3", ls="-", label="qy_k2")
                    ax.plot(t_eval, qx_k3, color="C4", ls="-", label="qx_k3")
                    ax.plot(t_eval, qy_k3, color="C5", ls="-", label="qy_k3")
                    ax.plot(t_eval, qx_k4, color="C6", ls="-", label="qx_k4")
                    ax.plot(t_eval, qy_k4, color="C7", ls="-", label="qy_k4")
                    ax.plot(t_eval, qx_k5, color="C8", ls="-", label="qx_k5")
                    ax.plot(t_eval, qy_k5, color="C9", ls="-", label="qy_k5")
                    ax.plot(t_eval, vx_m1, color="C10", ls="-", label="vx_m1")
                    ax.plot(t_eval, vy_m1, color="C11", ls="-", label="vy_m2")
                    ax.plot(t_eval, vx_m2, color="C12", ls="-", label="vx_m1")
                    ax.plot(t_eval, vy_m2, color="C13", ls="-", label="vy_m2")
                    ax.legend(prop={"size": 8})
                    plt.savefig(f"{base_dir}/{self.dataset_name}/{phase}_{itr}.png", dpi=100)
                    plt.close()

        return data_train, data_test

    def _get_initial_states(self, trials):
        qx_k1 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_qx_k1  # coordinate of spring 1
        qy_k1 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_qy_k1 + self.params.l1  # coordinate  of spring 1
        qx_k2 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_qx_k2  # coordinate  of spring 2
        qy_k2 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_qy_k2 + self.params.l1  # coordinate  of spring 2
        qx_k3 = qx_k2 - qx_k1 + self.params.l3  # coordinate  of spring 3
        qy_k3 = qy_k2 - qy_k1  # coordinate  of spring 3
        qx_k4 = qx_k2 + self.params.l3  # coordinate  of spring 4
        qy_k4 = qy_k2  # coordinate  of spring 4
        qx_k5 = qx_k1 - self.params.l3  # coordinate  of spring 5
        qy_k5 = qy_k1  # coordinate  of spring 5
        vx_m1 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_vx_m1  # velocity of mass 1
        vy_m1 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_vy_m1  # velocity of mass 1
        vx_m2 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_vx_m2  # velocity of mass 2
        vy_m2 = np.random.uniform(-1, 1, trials) * self.params.initial_dist_var_vy_m2  # velocity of mass 2
        return np.stack([qx_k1, qy_k1, qx_k2, qy_k2, qx_k3, qy_k3, qx_k4, qy_k4, qx_k5, qy_k5, vx_m1, vy_m1, vx_m2, vy_m2], axis=1)

    def _ode(self, t, state):
        q_k1, q_k2, q_k3, q_k4, q_k5, v_m1, v_m2 = np.split(state.T, 7)  # (14, 1000)
        m1, m2, k1, k2, k3, k4, k5, l1, l3, l4 = self.params.m1, self.params.m2, self.params.k1, self.params.k2, self.params.k3, self.params.k4, self.params.k5, self.params.l1, self.params.l3, self.params.l4
        q1dot = v_m1
        q2dot = v_m2
        q3dot = q2dot - q1dot
        q4dot = q2dot
        q5dot = q1dot
        s1 = -l1 + np.sqrt(np.sum(np.square(q_k1), axis=0))  # extention of k1
        s2 = -l1 + np.sqrt(np.sum(np.square(q_k2), axis=0))  # extention of k2
        s3 = -l3 + np.sqrt(np.sum(np.square(q_k3), axis=0))  # extention of k3
        s4 = -l4 + np.sqrt(np.sum(np.square(q_k4), axis=0))  # extention of k4
        s5 = -l4 + np.sqrt(np.sum(np.square(q_k5), axis=0))  # extention of k5
        v1dot = np.array([-k1(s1) * q_k1[0] / (s1 + l1) + k3(s3) * q_k3[0] / (s3 + l3) - k5(s5) * q_k5[0] / (s5 + l4), -k1(s1) * q_k1[1] / (s1 + l1) + k3(s3) * q_k3[1] / (s3 + l3) - k5(s5) * q_k5[1] / (s5 + l4)]) / m1
        v2dot = np.array([-k2(s2) * q_k2[0] / (s2 + l1) - k3(s3) * q_k3[0] / (s3 + l3) - k4(s4) * q_k4[0] / (s4 + l4), -k2(s2) * q_k2[1] / (s2 + l1) - k3(s3) * q_k3[1] / (s3 + l3) - k4(s4) * q_k4[1] / (s4 + l4)]) / m2
        return np.concatenate([q1dot, q2dot, q3dot, q4dot, q5dot, v1dot, v2dot], axis=0).T

    def _make_orbits(self, trials, n_steps, atol=1e-7, rtol=1e-9, **kwargs):
        initial_state = self._get_initial_states(trials)

        t_eval = np.arange(-self.params.initial_skip, n_steps + 1) * self.params.dt
        t_span = [t_eval[0], t_eval[-1]]

        orbits = scipy.integrate.solve_ivp(fun=lambda t, state: self._ode(t, state.reshape(initial_state.shape)).flatten(), t_span=t_span, y0=initial_state.flatten(), t_eval=t_eval, atol=atol, rtol=rtol, **kwargs)
        assert orbits.success

        u = orbits.y.reshape(initial_state.shape[0], initial_state.shape[1], self.params.initial_skip + n_steps + 1).transpose([0, 2, 1])

        u[..., 1] -= self.params.l1
        u[..., 3] -= self.params.l1
        u[..., 4] -= self.params.l3
        u[..., 6] -= self.params.l3
        u[..., 7] -= self.params.l1
        u[..., 8] += self.params.l3
        u[..., 9] -= self.params.l1

        u = u[:, self.params.initial_skip :]
        t_eval = t_eval[self.params.initial_skip :]

        data = argparse.Namespace(
            u=u,
            t_eval=t_eval,
            n_trials=u.shape[0],
            n_steps=u.shape[1],
            initial_state=initial_state,
        )

        return data
