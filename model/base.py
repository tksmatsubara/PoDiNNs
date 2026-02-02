import torch
import torch.nn as nn
import torch.optim
import time

# from torchdiffeq import odeint
import os

os.environ["DES_BACKEND"] = "torch"
import desolver


def get_activation_by_name(name):
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "leakyrelu":
        return nn.LeakyReLU
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "softplus":
        return nn.Softplus
    if name == "leakyrelu":
        return nn.LeakyReLU
    raise NotImplementedError


def get_MLP(input_dim, hidden_dim, output_dim, act="tanh", n_layers=3):
    assert n_layers >= 2
    Act = get_activation_by_name(act)
    sequence = [nn.Linear(input_dim, hidden_dim), Act()]
    for _ in range(n_layers - 2):
        sequence += [nn.Linear(hidden_dim, hidden_dim), Act()]
    sequence += [nn.Linear(hidden_dim, output_dim)]
    model = nn.Sequential(*sequence)
    return model


def get_GroupMLP(input_dim, hidden_dim, output_dim, act="tanh", n_layers=3, n_groups=1):
    assert n_layers >= 2
    Act = get_activation_by_name(act)
    sequence = [nn.Unflatten(-1, (-1, 1)), nn.Conv1d(input_dim * n_groups, hidden_dim * n_groups, kernel_size=1, groups=n_groups), Act()]
    for _ in range(n_layers - 2):
        sequence += [nn.Conv1d(hidden_dim * n_groups, hidden_dim * n_groups, kernel_size=1, groups=n_groups), Act()]
    sequence += [nn.Conv1d(hidden_dim * n_groups, output_dim * n_groups, kernel_size=1, groups=n_groups), nn.Flatten(-2, -1)]
    model = nn.Sequential(*sequence)
    return model


class RealNVP(nn.Module):

    def __init__(self, input_dim, half_input_dim, hidden_dim, act="tanh", n_layers=3, n_coupling=4):
        super(RealNVP, self).__init__()
        self.half_input_dim = half_input_dim
        sequence = []
        sequence.append(
            get_MLP(
                input_dim=half_input_dim,
                hidden_dim=hidden_dim,
                output_dim=(input_dim - half_input_dim) * 2,
                act=act,
                n_layers=n_layers,
            )
        )
        for itr in range(1, n_coupling):
            half_input_dim = input_dim - half_input_dim
            sequence.append(
                get_MLP(
                    input_dim=half_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=(input_dim - half_input_dim) * 2,
                    act=act,
                    n_layers=n_layers,
                )
            )
        self.net = nn.ModuleList(sequence)

    def forward(self, x):
        x1 = x[..., : self.half_input_dim]
        x2 = x[..., self.half_input_dim :]
        for layer in self.net:
            s, t = layer(x1).chunk(2, dim=-1)
            x2 = (x2 + t) * s.exp()
            x1, x2 = x2, x1
        return torch.concat([x1, x2], dim=-1)

    def reverse(self, x):
        x1 = x[..., : self.half_input_dim]
        x2 = x[..., self.half_input_dim :]
        for layer in list(self.net)[::-1]:
            x1, x2 = x2, x1
            s, t = layer(x1).chunk(2, dim=-1)
            x2 = x2 * (-s).exp() - t
        return torch.concat([x1, x2], dim=-1)


def get_Flow(*args, **kwargs):
    return RealNVP(*args, **kwargs)


def get_trainer(dataset, args):
    model_names = {
        "linear": LinearModel,  # Linear model.
        "node": NeuralODE,  # Neural ODE with the external position/velocity/force
        "node2": NeuralODE2nd,  # 2nd-order Neural ODE (the latter half is the velocity) with the external position/velocity/force
        "portnn": PortHNN,  # Port-Hamiltonian neural network with the external position as the input to the potential energy
        "podinn": PoDiNN,  # Proposed Poisson-Dirac neural network
        "podinnabs": PoDiNNAbs,  # Proposed Poisson-Dirac neural network for the absolute position with external position
    }
    model_names.update({k.lower(): v for k, v in globals().items() if isinstance(v, type) and issubclass(v, Trainer)})
    return model_names[args.model](dataset, args)


class Trainer(nn.Module):
    def __init__(self, dataset, args):
        super(Trainer, self).__init__()
        params = dataset.params
        # dataset.params.n_obs = params.n_obs
        # args.aux_dim = params.n_s + params.n_b
        self.args = args
        self.dataset = dataset
        self.n_itr = 0

        # prepare data
        self.t_eval_train = torch.from_numpy(self.dataset.data_train.t_eval).to(dtype=args.dtype, device=args.device)
        self.t_eval_test = torch.from_numpy(self.dataset.data_test.t_eval).to(dtype=args.dtype, device=args.device)
        self.dt = self.dataset.params.dt

        # batch, time, state -> time, batch, state
        self.u_train = torch.from_numpy(self.dataset.data_train.u.transpose(1, 0, 2)).to(dtype=args.dtype, device=args.device)
        self.u_test = torch.from_numpy(self.dataset.data_test.u.transpose(1, 0, 2)).to(dtype=args.dtype, device=args.device)
        self.state_scale = self.u_train.reshape(-1, self.u_train.shape[-1]).std(dim=0)

        # get networks and optimizers
        self.initialize_nets()
        self.optim = torch.optim.Adam(self.parameters(), args.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.args.total_steps)

    def initialize_nets(self):
        pass

    def train(self, stats):
        print(f"[TRAIN] running...")
        time_clock = time.time()
        stats["loss_train_prediction"] = []

        for itr in range(self.args.total_steps):
            n_steps, n_solutions, n_states = self.u_train.shape
            batch_size = self.args.batch_size
            device = self.args.device
            prediction_step = torch.randint(self.args.max_prediction_step, (1,)).item() + 1
            assert isinstance(prediction_step, int)

            # sample data
            idx_step = torch.randint(n_steps - prediction_step, (batch_size,), device=device)
            idx_solution = torch.randint(n_solutions, (batch_size,), device=device)
            t_observed = self.t_eval_train[idx_step]
            u_observed = self.u_train[idx_step, idx_solution]
            u_next = torch.stack([self.u_train[idx_step + 1 + itr, idx_solution] for itr in range(prediction_step)], dim=0)

            # prediction and loss
            t_eval = (torch.arange(prediction_step + 1) * self.dt).to(u_observed)
            with torch.enable_grad():
                u_predicted = self.prediction(t_observed, idx_solution, u_observed, t_eval, train=True)[1:]
                loss_prediction = ((u_next - u_predicted) / self.state_scale).__pow__(2).mean()

            # update
            self.optim.zero_grad()
            loss_prediction.backward()
            self.optim.step()
            self.scheduler.step()

            # logging
            stats["loss_train_prediction"].append(loss_prediction.item())
            if itr % self.args.log_freq == 0:
                # if hasattr(self, "print"):
                # self.print()
                time_clock_new = time.time()
                lr = self.optim.param_groups[0]["lr"]
                print(f"[TRAIN] itr {itr:05d}, time {time_clock_new - time_clock:.3f}s, loss {loss_prediction.item():.4e} lr {lr}")
                time_clock = time_clock_new

    def test(self, stats):
        print(f"[TEST] running...")
        time_clock = time.time()

        n_steps, n_solutions, n_dim = self.u_test.shape
        device = self.args.device

        # sample data
        idx_step = torch.zeros(n_solutions, device=device, dtype=torch.int)
        idx_solution = torch.arange(n_solutions, device=device)
        t_observed = self.t_eval_test[idx_step]
        u_observed = self.u_test[0]
        u_truth = self.u_test

        # prediction and loss
        u_predicted = self.prediction(t_observed, idx_solution, u_observed, self.t_eval_test, train=False)
        assert isinstance(u_predicted, torch.Tensor)
        loss_prediction = ((u_truth - u_predicted) / self.state_scale).__pow__(2).mean()

        # logging
        stats["loss_test_prediction"] = [loss_prediction.item()]
        print(f"[TEST], time {time.time() - time_clock:.3f}s, loss_prediction {loss_prediction.item():.4e}")
        stats["u_truth"] = u_truth.detach().cpu().numpy()
        stats["u_predicted"] = u_predicted.detach().cpu().numpy()
        stats["t_eval"] = self.t_eval_test.detach().cpu().numpy()

    def prediction(self, t_observed, idx_solution, u_observed, t_eval, train):
        # return odeint(
        #     lambda t, x: self.time_derivative(t, x, t_observed, idx_solution, train=train),
        #     u_observed,
        #     t_eval,
        #     method="dopri5",
        #     atol=1e-7,
        #     rtol=1e-9,
        # )
        ode = desolver.OdeSystem(
            lambda t, x: self.time_derivative(t, x, t_observed, idx_solution, train=train),
            y0=u_observed,
            dense_output=True,
            t=torch.concat([t_eval[:1], t_eval[-1:]], dim=0),
            dt=t_eval[1],
            rtol=1e-9,
            atol=1e-7,
            constants=dict(),
        )
        ode.method = self.args.solver
        ode.integrate(eta=False)
        if len(t_eval) == len(ode.y) == 2:
            return torch.stack(ode.y, dim=0)  # type: ignore
        assert ode.sol is not None
        return ode.sol(t_eval)


class LinearModel(Trainer):

    def initialize_nets(self):
        params = self.dataset.params
        self.mat = nn.Linear(params.n_obs + params.n_b * 2 + params.n_s, params.n_obs + params.n_b * 2 + params.n_s, bias=True)

    def time_derivative(self, t, x, t0, i, train=True):
        v_b = self.dataset.external_flow(t + t0, i, train=train)
        q_b = self.dataset.external_position(t + t0, i, train=train)
        f_s = self.dataset.external_effort(t + t0, i, train=train)
        return self.mat(torch.concat([x, q_b, v_b, f_s], dim=-1))


class NeuralODE(Trainer):

    def __init__(self, dataset, args):
        super(NeuralODE, self).__init__(dataset, args)

    def initialize_nets(self):
        params = self.dataset.params
        self.net = get_MLP(input_dim=params.n_k + params.n_m + params.n_b * 2 + params.n_s, hidden_dim=self.args.hidden_dim, output_dim=params.n_k + params.n_m, act=self.args.act, n_layers=self.args.n_layers)

    def time_derivative(self, t, x, t0, i, train=True):
        v_b = self.dataset.external_flow(t + t0, i, train=train)
        q_b = self.dataset.external_position(t + t0, i, train=train)
        f_s = self.dataset.external_effort(t + t0, i, train=train)
        return self.net(torch.concat([x, q_b, v_b, f_s], dim=-1))


class NeuralODE2nd(NeuralODE):

    def initialize_nets(self):
        params = self.dataset.params
        self.net = get_MLP(input_dim=params.n_k + params.n_m + params.n_b * 2 + params.n_s, hidden_dim=self.args.hidden_dim, output_dim=params.n_k, act=self.args.act, n_layers=self.args.n_layers)

    def time_derivative(self, t, x, t0, i, train=True):
        _, v = x.chunk(2, dim=-1)
        v_b = self.dataset.external_flow(t + t0, i, train=train)
        q_b = self.dataset.external_position(t + t0, i, train=train)
        f_s = self.dataset.external_effort(t + t0, i, train=train)
        vdot = self.net(torch.concat([x, q_b, v_b, f_s], dim=-1))
        return torch.concat([v, vdot], dim=-1)


class PortHNN(Trainer):

    def __init__(self, dataset, args):
        super(PortHNN, self).__init__(dataset, args)

    def initialize_nets(self):
        super(PortHNN, self).initialize_nets()
        params = self.dataset.params
        assert params.n_k == params.n_m
        self.n_conf = params.n_k

        self.potential_energy = get_MLP(input_dim=self.n_conf + params.n_b, hidden_dim=self.args.hidden_dim, output_dim=1, act=self.args.act, n_layers=self.args.n_layers)
        self.mass_log10 = nn.Parameter(torch.zeros(self.n_conf))
        self.D = get_MLP(input_dim=self.n_conf * 2 + params.n_b, hidden_dim=self.args.hidden_dim, output_dim=self.n_conf**2, act=self.args.act, n_layers=self.args.n_layers)
        if params.n_s > 0:
            self.g = get_MLP(input_dim=self.n_conf + params.n_b, hidden_dim=self.args.hidden_dim, output_dim=self.n_conf * params.n_s, act=self.args.act, n_layers=self.args.n_layers)
        else:
            self.g = None

    def time_derivative(self, t, x, t0, i, train=True):
        batch_size = x.shape[0]
        q, v = x.chunk(2, dim=-1)
        m = 10**self.mass_log10
        p = v * m
        q = q.requires_grad_(True)
        q_b = self.dataset.external_position(t + t0, i, train=train)
        f_s = self.dataset.external_effort(t + t0, i, train=train)
        with torch.enable_grad():
            q_ex = torch.concat([q, q_b], dim=-1)
            potential_energy = self.potential_energy(q_ex).sum()
        Hq = torch.autograd.grad(potential_energy, q, create_graph=True)[0]
        # not make sure symmetry because of the nonlinearlity
        Dq = self.D(torch.concat([q_ex, p], dim=-1)).reshape(batch_size, self.n_conf, self.n_conf)
        pdot = -Hq - torch.einsum("bij,bj->bi", Dq, v)
        if self.g is not None:
            gq = self.g(q_ex).reshape(batch_size, self.n_conf, self.dataset.params.n_s)
            pdot = pdot + torch.einsum("bij,bj->bi", gq, f_s)
        vdot = pdot / m
        return torch.concat([v, vdot], dim=-1)


class PNN(Trainer):

    def __init__(self, dataset, args):
        self.n_obs = dataset.params.n_k + dataset.params.n_m
        self.n_obs_k = dataset.params.n_k
        self.n_obs_m = dataset.params.n_m
        self.n_real_conf = min(dataset.params.n_k, dataset.params.n_m)
        dataset.params.n_k = self.n_real_conf
        dataset.params.n_m = self.n_real_conf
        super(PNN, self).__init__(dataset, args)

    def initialize_nets(self):
        super(PNN, self).initialize_nets()
        params = self.dataset.params
        assert params.n_k == params.n_m
        self.n_conf = params.n_k

        self.energy = get_MLP(input_dim=self.n_conf * 2, hidden_dim=self.args.hidden_dim, output_dim=1, act=self.args.act, n_layers=self.args.n_layers)
        self.flow = get_Flow(input_dim=self.n_obs, half_input_dim=self.n_obs_k, hidden_dim=self.args.hidden_dim, act=self.args.act, n_layers=self.args.n_layers, n_coupling=4)

    def time_derivative(self, t, x, t0, i, train=True):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            energy = self.energy(x).sum()
        dH = torch.autograd.grad(energy, x, create_graph=train)[0]
        Hq, Hp = dH.chunk(2, dim=-1)
        qdot = Hp
        pdot = -Hq
        return torch.concat([qdot, pdot], dim=-1)

    def prediction(self, t_observed, idx_solution, u_observed, t_eval, train):
        z = self.flow(u_observed)
        u_hidden = z[..., -self.n_real_conf * 2 :]
        casimir = z[..., : -self.n_real_conf * 2]
        ode = desolver.OdeSystem(
            lambda t, x: self.time_derivative(t, x, t_observed, idx_solution, train=train),
            y0=u_hidden,
            dense_output=True,
            t=torch.concat([t_eval[:1], t_eval[-1:]], dim=0),
            dt=t_eval[1],
            rtol=1e-9,
            atol=1e-7,
            constants=dict(),
        )
        ode.method = self.args.solver
        ode.integrate(eta=False)
        if len(t_eval) == len(ode.y) == 2:
            u_predicted = torch.stack(ode.y, dim=0)  # type: ignore
        else:
            assert ode.sol is not None
            u_predicted = ode.sol(t_eval)
        casimir = casimir.repeat(len(t_eval), 1).reshape(len(t_eval), *casimir.shape)
        return torch.concat([casimir, u_predicted], dim=-1)


class Element(nn.Module):
    def __init__(self, n, hidden_dim, act, n_layers, combined=False, linear=False, input_dim=1):
        super(Element, self).__init__()
        assert n % input_dim == 0
        self.n = n
        self.linear = linear
        self.combined = combined
        if self.n == 0:
            return
        if self.linear:
            self.constant_log10 = nn.Parameter(torch.zeros(n // input_dim))
        elif self.combined or self.n == 1:
            self.net = get_MLP(input_dim=n, hidden_dim=hidden_dim, output_dim=1, act=act, n_layers=n_layers)
        else:
            self.net = get_GroupMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, act=act, n_layers=n_layers, n_groups=n // input_dim)

    def get_constant(self):
        return 10**self.constant_log10

    def forward(self, x):
        if self.n == 0:
            return torch.zeros([*x.shape[:-1], 0], device=x.device)
        if self.linear:
            return x.__pow__(2).__truediv__(2 * self.get_constant())
        return self.net(x)

    def grad(self, x):
        if self.linear:
            return x.__truediv__(self.get_constant())
        x = x.requires_grad_(True)
        with torch.enable_grad():
            potential = self(x).sum()
        return torch.autograd.grad(potential, x, create_graph=True)[0]


class PoDiNN(Trainer):

    def __init__(self, dataset, args):
        if args.set_d is not None:
            dataset.params.n_d = args.set_d
        if args.set_g is not None:
            dataset.params.n_g = args.set_g
        super(PoDiNN, self).__init__(dataset, args)

    def initialize_nets(self):
        params = self.dataset.params

        # k: spring, capacitor
        self.elastic_energies = Element(params.n_k, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=self.args.linear_C)
        # m: mass, inductor
        self.kinetic_energies = Element(params.n_m, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=self.args.linear_I)
        # d: damper, resistor
        self.damping = Element(params.n_d, self.args.hidden_dim, self.args.act, self.args.n_layers)
        # g: dual damper, resistor
        self.damping_dual = Element(params.n_g, self.args.hidden_dim, self.args.act, self.args.n_layers)

        # elements determining bivector and thus coupling
        # to determine the dissipative elements
        self.B_k2g = nn.Parameter((torch.rand([params.n_k, params.n_g]) * 2 - 1) * 0.1)
        self.B_m2d = nn.Parameter((torch.rand([params.n_m, params.n_d]) * 2 - 1) * 0.1)
        self.B_s2g = nn.Parameter((torch.rand([params.n_s, params.n_g]) * 2 - 1) * 0.1)
        self.B_b2d = nn.Parameter((torch.rand([params.n_b, params.n_d]) * 2 - 1) * 0.1)
        # to determine the time derivative
        self.B_m2k = nn.Parameter(torch.zeros([params.n_m, params.n_k]))
        self.B_b2k = nn.Parameter(torch.zeros([params.n_b, params.n_k]))
        self.B_s2m = nn.Parameter(torch.zeros([params.n_s, params.n_m]))

    def time_derivative(self, t, x, t0, i, train=True):
        params = self.dataset.params
        batchsize, n_states = x.shape
        assert n_states == params.n_k + params.n_m
        q_k = x[..., : params.n_k]
        v_m = x[..., params.n_k : params.n_k + params.n_m]  # or p_m

        if self.elastic_energies.linear:  # given as voltage
            c = self.elastic_energies.get_constant()
            e_k = q_k
        else:  # given as displacement or electric charge
            c = 1.0
            e_k = self.elastic_energies.grad(q_k)  # Hq=-k q_m

        if self.kinetic_energies.linear:  # given as velocity or current
            m = self.kinetic_energies.get_constant()
            e_m = v_m
        else:  # given as momentum or magnetic flux
            m = 1.0
            e_m = self.kinetic_energies.grad(v_m)  # Hp=v_m

        e_s = self.dataset.external_effort(t + t0, i, train=train)
        e_b = self.dataset.external_flow(t + t0, i, train=train)

        f_d = e_m @ self.B_m2d + e_b @ self.B_b2d
        f_g = e_k @ self.B_k2g + e_s @ self.B_s2g

        e_d = self.damping(f_d)
        e_g = self.damping_dual(f_g)

        f_k = e_m @ self.B_m2k + e_b @ self.B_b2k - e_g @ self.B_k2g.t()
        f_m = -e_k @ self.B_m2k.t() + e_s @ self.B_s2m - e_d @ self.B_m2d.t()

        return torch.cat([f_k / c, f_m / m], dim=-1)

    def print(self):
        if self.B_k2g.numel() > 0:
            print("B_k2g:\n", self.B_k2g.cpu().numpy())
        if self.B_m2d.numel() > 0:
            print("B_m2d:\n", self.B_m2d.cpu().numpy())
        if self.B_s2g.numel() > 0:
            print("B_s2g:\n", self.B_s2g.cpu().numpy())
        if self.B_b2d.numel() > 0:
            print("B_b2d:\n", self.B_b2d.cpu().numpy())
        # to determine the time derivative
        if self.B_m2k.numel() > 0:
            print("B_m2k:\n", self.B_m2k.cpu().numpy())
        if self.B_b2k.numel() > 0:
            print("B_b2k:\n", self.B_b2k.cpu().numpy())
        if self.B_s2m.numel() > 0:
            print("B_s2m:\n", self.B_s2m.cpu().numpy())
        if self.elastic_energies.linear:
            print("c:", self.elastic_energies.get_constant().cpu().numpy())
        if self.kinetic_energies.linear:
            print("m:", self.kinetic_energies.get_constant().cpu().numpy())


class PoDiNNAbs(Trainer):

    def __init__(self, dataset, args):
        if args.set_d is not None:
            dataset.params.n_d = args.set_d
        if args.set_g is not None:
            dataset.params.n_g = args.set_g
        super(PoDiNNAbs, self).__init__(dataset, args)

    def initialize_nets(self):
        params = self.dataset.params
        # k: spring, capacitor
        self.elastic_energies = Element(
            params.n_k + params.n_b,
            self.args.hidden_dim,
            self.args.act,
            self.args.n_layers,
            combined=True,
            linear=self.args.linear_C,
        )
        # m: mass, inductor
        self.kinetic_energies = Element(
            params.n_m,
            self.args.hidden_dim,
            self.args.act,
            self.args.n_layers,
            linear=self.args.linear_I,
        )
        # d: damper, resistor
        self.damping = Element(params.n_d, self.args.hidden_dim, self.args.act, self.args.n_layers)
        # g: dual damper, resistor
        self.damping_dual = Element(params.n_g, self.args.hidden_dim, self.args.act, self.args.n_layers)

        # elements determining bivector and thus coupling
        # to determine the dissipative elements
        self.B_k2g = nn.Parameter((torch.rand([params.n_k, params.n_g]) * 2 - 1) * 0.1)
        self.B_m2d = nn.Parameter((torch.rand([params.n_m, params.n_d]) * 2 - 1) * 0.1)
        self.B_s2g = nn.Parameter((torch.rand([params.n_s, params.n_g]) * 2 - 1) * 0.1)
        self.B_b2d = nn.Parameter((torch.rand([params.n_b, params.n_d]) * 2 - 1) * 0.1)  # boundary as part of position
        # to determine the time derivative
        # self.B_m2k = nn.Parameter(torch.zeros([params.n_m, params.n_k]))
        self.B_m2k_el = nn.Parameter(torch.zeros([1]))
        self.B_b2k = nn.Parameter(torch.zeros([params.n_b, params.n_k]))
        self.B_s2m = nn.Parameter(torch.zeros([params.n_s, params.n_m]))

    @property
    def B_m2k(self):
        return torch.eye(self.dataset.params.n_k, device=self.B_m2k_el.device) * self.B_m2k_el

    def time_derivative(self, t, x, t0, i, train=True):
        params = self.dataset.params
        batchsize, n_states = x.shape
        assert n_states == params.n_k + params.n_m
        assert self.kinetic_energies.linear
        m = self.kinetic_energies.get_constant()

        q_k = x[..., : params.n_k]
        v_m = x[..., params.n_k : params.n_k + params.n_m]
        # p_m = v_m * m

        q_b = self.dataset.external_position(t + t0, i, train=train)
        e_b = self.dataset.external_flow(t + t0, i, train=train)  # damper may be connected to boundary
        e_s = self.dataset.external_effort(t + t0, i, train=train)

        e_k = self.elastic_energies.grad(torch.concat([q_k, q_b], dim=-1))[..., : params.n_k]
        # e_m = self.kinetic_energies.grad(p_m)
        e_m = v_m
        f_d = e_m @ self.B_m2d + e_b @ self.B_b2d
        f_g = e_k @ self.B_k2g + e_s @ self.B_s2g

        e_d = self.damping(f_d)
        e_g = self.damping_dual(f_g)

        f_k = e_m @ self.B_m2k + e_b @ self.B_b2k - e_g @ self.B_k2g.t()
        f_m = -e_k @ self.B_m2k.t() + e_s @ self.B_s2m - e_d @ self.B_m2d.t()

        return torch.cat([f_k, f_m / m], dim=-1)

    def print(self):
        if self.B_k2g.numel() > 0:
            print("B_k2g:\n", self.B_k2g.cpu().numpy())
        if self.B_m2d.numel() > 0:
            print("B_m2d:\n", self.B_m2d.cpu().numpy())
        if self.B_s2g.numel() > 0:
            print("B_s2g:\n", self.B_s2g.cpu().numpy())
        # to determine the time derivative
        if self.B_m2k.numel() > 0:
            print("B_m2k:\n", self.B_m2k.cpu().numpy())
        if self.B_s2m.numel() > 0:
            print("B_s2m:\n", self.B_s2m.cpu().numpy())
        print("m:", self.kinetic_energies.get_constant().cpu().numpy())


class PoDiNN2Dim(Trainer):
    def __init__(self, dataset, args):
        if args.set_d is not None:
            dataset.params.n_d = args.set_d
        if args.set_g is not None:
            dataset.params.n_g = args.set_g
        super(PoDiNN2Dim, self).__init__(dataset, args)

    def initialize_nets(self):
        params = self.dataset.params

        # k: spring, capacitor
        self.elastic_energies = Element(params.n_k, self.args.hidden_dim, self.args.act, self.args.n_layers, input_dim=2, linear=self.args.linear_C)
        # m: mass, inductor
        self.kinetic_energies = Element(params.n_m, self.args.hidden_dim, self.args.act, self.args.n_layers, input_dim=2, linear=self.args.linear_I)
        # d: damper, resistor
        self.damping = Element(params.n_d, self.args.hidden_dim, self.args.act, self.args.n_layers)
        # g: dual damper, resistor
        self.damping_dual = Element(params.n_g, self.args.hidden_dim, self.args.act, self.args.n_layers)

        # # elements determining bivector and thus coupling
        # # to determined the dissipative elements
        self.B_k2g = nn.Parameter((torch.rand([params.n_k, params.n_g]) * 2 - 1) * 0.1)
        self.B_m2d = nn.Parameter((torch.rand([params.n_m, params.n_d]) * 2 - 1) * 0.1)
        self.B_s2g = nn.Parameter((torch.rand([params.n_s, params.n_g]) * 2 - 1) * 0.1)
        self.B_b2d = nn.Parameter((torch.rand([params.n_b, params.n_d]) * 2 - 1) * 0.1)
        # to determined the time derivative
        self.B_m2k = nn.Parameter(torch.zeros([params.n_m, params.n_k]))
        self.B_b2k = nn.Parameter(torch.zeros([params.n_b, params.n_k]))
        self.B_s2m = nn.Parameter(torch.zeros([params.n_s, params.n_m]))

    def time_derivative(self, t, x, t0, i, train=True):
        params = self.dataset.params
        assert x.shape[-1] == params.n_k + params.n_m
        assert self.kinetic_energies.linear
        m = self.kinetic_energies.get_constant()

        q_k = x[..., : params.n_k]
        v_m = x[..., params.n_k :]
        # p_m = v_m * m

        e_k = self.elastic_energies.grad(q_k)  # Hq=-k q_m
        # e_m = self.kinetic_energies.grad(p_m)  # Hp=v_m
        e_m = v_m

        e_s = self.dataset.external_effort(t + t0, i, train=train)
        e_b = self.dataset.external_flow(t + t0, i, train=train)

        f_d = e_m @ self.B_m2d + e_b @ self.B_b2d
        f_g = e_k @ self.B_k2g + e_s @ self.B_s2g

        e_d = self.damping(f_d)
        e_g = self.damping_dual(f_g)

        f_k = e_m @ self.B_m2k + e_b @ self.B_b2k - e_g @ self.B_k2g.t()
        f_m = -e_k @ self.B_m2k.t() + e_s @ self.B_s2m - e_d @ self.B_m2d.t()
        return torch.cat([f_k, f_m / torch.repeat_interleave(m, 2)], dim=-1)

    def print(self):
        if self.B_k2g.numel() > 0:
            print("B_k2g:\n", self.B_k2g.cpu().numpy())
        if self.B_m2d.numel() > 0:
            print("B_m2d:\n", self.B_m2d.cpu().numpy())
        if self.B_s2g.numel() > 0:
            print("B_s2g:\n", self.B_s2g.cpu().numpy())
        if self.B_b2d.numel() > 0:
            print("B_b2d:\n", self.B_b2d.cpu().numpy())
        # to determined the time derivative
        if self.B_m2k.numel() > 0:
            print("B_m2k:\n", self.B_m2k.cpu().numpy())
        if self.B_b2k.numel() > 0:
            print("B_b2k:\n", self.B_b2k.cpu().numpy())
        if self.B_s2m.numel() > 0:
            print("B_s2m:\n", self.B_s2m.cpu().numpy())
        print("m:", self.kinetic_energies.get_constant().cpu().numpy())


class PoDiNNDC(Trainer):

    def __init__(self, dataset, args):
        assert args.set_d is None
        assert args.set_g is None
        assert not args.linear_C
        assert not args.linear_I
        super(PoDiNNDC, self).__init__(dataset, args)

    def initialize_nets(self):
        params = self.dataset.params

        # domain 1: assumed mechanical or rotational
        self.elastic_energies1 = Element(params.n_k1, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=False)
        self.kinetic_energies1 = Element(params.n_m1, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=True)
        self.damping1 = Element(params.n_d1, self.args.hidden_dim, self.args.act, self.args.n_layers)
        self.damping_dual1 = Element(params.n_g1, self.args.hidden_dim, self.args.act, self.args.n_layers)
        self.B_k2g1 = nn.Parameter((torch.rand([params.n_k1, params.n_g1]) * 2 - 1) * 0.1)
        # self.B_m2d1 = nn.Parameter((torch.rand([params.n_m1, params.n_d1]) * 2 - 1) * 0.1)
        self.B_m2d1 = nn.Parameter(torch.ones([params.n_m1, params.n_d1]), requires_grad=False)
        self.B_s2g1 = nn.Parameter((torch.rand([params.n_s1, params.n_g1]) * 2 - 1) * 0.1)
        self.B_b2d1 = nn.Parameter((torch.rand([params.n_b1, params.n_d1]) * 2 - 1) * 0.1)
        # self.B_m2k1 = nn.Parameter(torch.zeros([params.n_m1, params.n_k1]))
        self.B_m2k1 = nn.Parameter(torch.ones([params.n_m1, params.n_k1]), requires_grad=False)
        self.B_b2k1 = nn.Parameter(torch.zeros([params.n_b1, params.n_k1]))
        self.B_s2m1 = nn.Parameter(torch.zeros([params.n_s1, params.n_m1]))
        # domain 2: assumed electro-magnetic domain
        # self.elastic_energies2 = Element(params.n_k2, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=self.args.linear_C)
        self.elastic_energies2 = Element(params.n_k2, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=True)
        self.kinetic_energies2 = Element(params.n_m2, self.args.hidden_dim, self.args.act, self.args.n_layers, linear=True)
        self.damping2 = Element(params.n_d2, self.args.hidden_dim, self.args.act, self.args.n_layers)
        self.damping_dual2 = Element(params.n_g2, self.args.hidden_dim, self.args.act, self.args.n_layers)
        self.B_k2g2 = nn.Parameter((torch.rand([params.n_k2, params.n_g2]) * 2 - 1) * 0.1)
        # self.B_m2d2 = nn.Parameter((torch.rand([params.n_m2, params.n_d2]) * 2 - 1) * 0.1)
        self.B_m2d2 = nn.Parameter(torch.ones([params.n_m2, params.n_d2]), requires_grad=False)
        self.B_s2g2 = nn.Parameter((torch.rand([params.n_s2, params.n_g2]) * 2 - 1) * 0.1)
        self.B_b2d2 = nn.Parameter((torch.rand([params.n_b2, params.n_d2]) * 2 - 1) * 0.1)
        self.B_m2k2 = nn.Parameter(torch.zeros([params.n_m2, params.n_k2]))
        self.B_b2k2 = nn.Parameter(torch.zeros([params.n_b2, params.n_k2]))
        # self.B_s2m2 = nn.Parameter(torch.zeros([params.n_s2, params.n_m2]))
        self.B_s2m2 = nn.Parameter(torch.ones([params.n_s2, params.n_m2]), requires_grad=False)

        # cross-domain factor
        self.B_m2_m1 = nn.Parameter(torch.zeros([params.n_m1, params.n_m2]))
        #  TODO: all possible couplings
        # self.B_m1_g2 = nn.Parameter(torch.zeros([params.n_m1, params.n_g2]))
        # self.B_m1_f2 = nn.Parameter(torch.zeros([params.n_m1, params.n_f2]))
        # self.B_g1_m2 = nn.Parameter(torch.zeros([params.n_g1, params.n_m2]))
        # self.B_f1_m2 = nn.Parameter(torch.zeros([params.n_f1, params.n_m2]))

    def time_derivative(self, t, x, t0, i, train=True):

        params = self.dataset.params
        assert x.shape[-1] == params.n_k1 + params.n_m1 + params.n_k2 + params.n_m2
        assert self.kinetic_energies1.linear
        assert self.elastic_energies2.linear
        assert self.kinetic_energies2.linear

        c1 = 1.0
        c2 = self.elastic_energies2.get_constant() if self.elastic_energies2.n > 0 else 1.0
        m1 = self.kinetic_energies1.get_constant() if self.kinetic_energies1.n > 0 else 1.0
        m2 = self.kinetic_energies2.get_constant() if self.kinetic_energies2.n > 0 else 1.0

        q_k1 = x[..., : params.n_k1]
        v_m1 = x[..., params.n_k1 : params.n_k1 + params.n_m1]
        q_k2 = x[..., params.n_k1 + params.n_m1 : params.n_k1 + params.n_m1 + params.n_k2]
        v_m2 = x[..., params.n_k1 + params.n_m1 + params.n_k2 : params.n_k1 + params.n_m1 + params.n_k2 + params.n_m2]

        e_k1 = self.elastic_energies1.grad(q_k1)
        e_k2 = q_k2
        e_m1 = v_m1
        e_m2 = v_m2

        e_s = self.dataset.external_effort(t + t0, i, train=train)
        e_s1 = e_s[..., : params.n_s1]
        e_s2 = e_s[..., params.n_s1 :]
        e_b = self.dataset.external_flow(t + t0, i, train=train)
        e_b1 = e_b[..., : params.n_b1]
        e_b2 = e_b[..., params.n_b1 :]

        f_d1 = e_m1 @ self.B_m2d1 + e_b1 @ self.B_b2d1
        f_d2 = e_m2 @ self.B_m2d2 + e_b2 @ self.B_b2d2
        f_g1 = e_k1 @ self.B_k2g1 + e_s1 @ self.B_s2g1
        f_g2 = e_k2 @ self.B_k2g2 + e_s2 @ self.B_s2g2

        e_d1 = self.damping1(f_d1)
        e_d2 = self.damping2(f_d2)
        e_g1 = self.damping_dual1(f_g1)
        e_g2 = self.damping_dual2(f_g2)
        f_k1 = e_m1 @ self.B_m2k1 + e_b1 @ self.B_b2k1 - e_g1 @ self.B_k2g1.t()
        f_k2 = e_m2 @ self.B_m2k2 + e_b2 @ self.B_b2k2 - e_g2 @ self.B_k2g2.t()
        f_m1 = -e_k1 @ self.B_m2k1.t() + e_s1 @ self.B_s2m1 - e_d1 @ self.B_m2d1.t() + e_m2 @ self.B_m2_m1
        f_m2 = -e_k2 @ self.B_m2k2.t() + e_s2 @ self.B_s2m2 - e_d2 @ self.B_m2d2.t() - e_m1 @ self.B_m2_m1.t()

        return torch.cat([f_k1 / c1, f_m1 / m1, f_k2 / c2, f_m2 / m2], dim=-1)

    def print(self):
        if self.B_k2g1.numel() > 0:
            print("B_k2g1:\n", self.B_k2g1.detach().cpu().numpy())
        if self.B_m2d1.numel() > 0:
            print("B_m2d1:\n", self.B_m2d1.detach().cpu().numpy())
        if self.B_s2g1.numel() > 0:
            print("B_s2g1:\n", self.B_s2g1.detach().cpu().numpy())
        if self.B_b2d1.numel() > 0:
            print("B_b2d1:\n", self.B_b2d1.detach().cpu().numpy())
        # to determined the time derivative
        if self.B_m2k1.numel() > 0:
            print("B_m2k1:\n", self.B_m2k1.detach().cpu().numpy())
        if self.B_b2k1.numel() > 0:
            print("B_b2k1:\n", self.B_b2k1.detach().cpu().numpy())
        if self.B_s2m1.numel() > 0:
            print("B_s2m1:\n", self.B_s2m1.detach().cpu().numpy())
        if self.elastic_energies1.n > 0 and self.elastic_energies1.linear:
            print("c1:", self.elastic_energies1.get_constant().detach().cpu().numpy())
        if self.kinetic_energies1.n > 0 and self.kinetic_energies1.linear:
            print("m1:", self.kinetic_energies1.get_constant().detach().cpu().numpy())
        if self.B_k2g2.numel() > 0:
            print("B_k2g2:\n", self.B_k2g2.detach().cpu().numpy())
        if self.B_m2d2.numel() > 0:
            print("B_m2d2:\n", self.B_m2d2.detach().cpu().numpy())
        if self.B_s2g2.numel() > 0:
            print("B_s2g2:\n", self.B_s2g2.detach().cpu().numpy())
        if self.B_b2d2.numel() > 0:
            print("B_b2d2:\n", self.B_b2d2.detach().cpu().numpy())
        # to determined the time derivative
        if self.B_m2k2.numel() > 0:
            print("B_m2k2:\n", self.B_m2k2.detach().cpu().numpy())
        if self.B_b2k2.numel() > 0:
            print("B_b2k2:\n", self.B_b2k2.detach().cpu().numpy())
        if self.B_s2m2.numel() > 0:
            print("B_s2m2:\n", self.B_s2m2.detach().cpu().numpy())
        if self.elastic_energies2.n > 0 and self.elastic_energies2.linear:
            print("c2:", self.elastic_energies2.get_constant().detach().cpu().numpy())
        if self.kinetic_energies2.n > 0 and self.kinetic_energies2.linear:
            print("m2:", self.kinetic_energies2.get_constant().detach().cpu().numpy())
        if self.B_m2_m1.numel() > 0:
            print("B_m2_m1:\n", self.B_m2_m1.detach().cpu().numpy())
        print("===========================")
