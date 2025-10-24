import torch
import numpy as np
import torch.nn as nn
from argparse import Namespace
import torch.nn.functional as F
import torch.distributions as D
from utils import draw_plot


class MixtureOfGaussians(nn.Module):
    def __init__(self, num_components, args: Namespace):
        super(MixtureOfGaussians, self).__init__()
        self.num_components = num_components
        self.input_dim = args.n_items

        # Learnable parameters
        # self.means = (torch.rand(num_components, args.n_items) / 100 + 0.5).to(args.device)
        self.means = torch.sigmoid(torch.rand(num_components, args.n_items)).to(args.device) * 0.1 + 0.45
        self.log_stds = (-torch.ones(num_components, args.n_items) * 3).to(args.device)
        self.weights = (torch.ones(num_components) / num_components).to(args.device)
        self.args = args

    def sample(self, num_samples):
        component_indices = torch.multinomial(F.softmax(self.weights, dim=0), num_samples, replacement=True)
        # component_indices: [n_revenue_samples]
        component_indices = component_indices.unsqueeze(-1).expand(-1, self.args.n_items)
        # component_indices: [n_revenue_samples, n_items]
        chosen_means = torch.gather(self.means, 0, component_indices)
        # chosen_means: [n_revenue_samples, n_items]
        chosen_stds = torch.exp(torch.gather(self.log_stds, 0, component_indices))
        # chosen_means: [n_revenue_samples, n_items]
        normal = D.Normal(chosen_means, chosen_stds)

        return normal.sample().detach().clone()

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num * (2 if input_dim > 100 else 1), bias=True)
        self.fc3 = nn.Linear(hidden_num * (2 if input_dim > 100 else 1), input_dim, bias=True)
        # self.act = lambda x: torch.tanh(x)

        self.input_dim = input_dim

        self.fct1 = nn.Linear(1, hidden_num, bias=True)
        self.fct2 = nn.Linear(hidden_num, 1, bias=True)

    def forward(self, z0, zt, t):
        # inputs = torch.cat([z0, t], dim=1)
        x = self.fc1(z0)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)

        sigma = self.fct1(t)
        sigma = torch.tanh(sigma)
        sigma = self.fct2(sigma)

        return x * zt * sigma, x, sigma.detach()
        # return torch.bmm(x, zt.unsqueeze(-1)).squeeze(-1) * sigma, x, sigma.detach()


class RectifiedFlow:
    def __init__(self, args):
        self.model = MLP(args.n_items, hidden_num=args.flow_nn_hidden).to(args.device)
        self.N = 1000
        self.args = args
        self.noise = MixtureOfGaussians(3, self.args)
        self.ones = torch.ones((1, 1)).to(self.args.device)

    def get_train_tuple(self, z0=None, z1=None):
        t = torch.rand((z0.shape[0], 1)).to(self.args.device)
        zt = t * z1 + (1 - t) * z0
        target = z1 - z0

        return zt, t, target

    def get_data(self):
        samples_0 = self.noise.sample(self.args.flow_train_bs)
        # samples_0 = torch.sigmoid(torch.randn(self.args.flow_train_bs, self.args.n_items)).to(self.args.device) * 0.1 + 0.45

        # To make sure the ordinary differentiable equation has a unique solution, each bundle (like [0, 1, 1]) is represented by a spherical Gaussian centered at it.
        # normal = D.Normal(torch.round(samples_0), 0.01)

        normal = D.Normal(torch.where(samples_0 < 0.5, torch.zeros_like(samples_0), torch.ones_like(samples_0)), 0.1)
        samples_1 = normal.sample()

        # samples_1 = torch.where(samples_0 < 0.5, torch.zeros_like(samples_0), torch.ones_like(samples_0))

        # print('Shape of the samples:', samples_0.shape, samples_1.shape)

        return samples_0, samples_1

    def load(self, path):
        loaded_model = MLP(self.args.n_items, hidden_num=self.args.flow_nn_hidden)
        loaded_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model = loaded_model.to(self.args.device)


    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        # Use the Euler method to sample from the learned flow
        if N is None:
            N = self.N

        dt = 1. / N
        traj = []  # Store trajectory

        z = z0.detach().clone()
        traj.append(z.detach().clone())
        sigma = 0

        for i in range(N):
            t = self.ones.repeat(z.shape[0], 1) * i / N
            pred, Q, _sigma = self.model(z0, z, t)
            # Q: [bs, n_items, n_items]  sigma: [bs, 1]
            sigma = sigma + _sigma
            z = z.detach().clone() + pred * dt

            traj.append(z.detach().clone())

        return traj, Q, sigma / N

    def sample_ode_grad(self, z0=None, N=None):
        # Use the Euler method to sample from the learned flow
        if N is None:
            N = self.N

        dt = 1. / N
        traj = []  # Store trajectory

        z = z0.detach().clone()
        traj.append(z.detach().clone())
        sigma = 0

        for i in range(N):
            t = self.ones.repeat(z.shape[0], 1) * i / N
            pred, Q, _sigma = self.model(z0, z, t)
            # Q: [bs, n_items, n_items]  sigma: [bs, 1]
            sigma = sigma + _sigma
            z = z.detach().clone() + pred * dt

            traj.append(z.detach().clone())

        return traj, Q, sigma / N

    def sample_ode_super_grad(self, z0=None, N=None):
        # Use the Euler method to sample from the learned flow
        if N is None:
            N = self.N

        dt = 1. / N
        traj = []  # Store trajectory

        z = z0.detach().clone()
        traj.append(z.detach().clone())
        sigma = 0

        for i in range(N):
            t = self.ones.repeat(z.shape[0], 1) * i / N
            pred, Q, _sigma = self.model(z0, z, t)
            # Q: [bs, n_items, n_items]  sigma: [bs, 1]
            sigma = sigma + _sigma
            z = z + pred * dt

            traj.append(z)

        return traj, Q, sigma / N

def train(flow, args):
    # Generate pairs

    optimizer = torch.optim.Adam(flow.model.parameters(), lr=5e-3)
    loss_curve = []

    print('Starting flow training...')
    for i in range(args.flow_train_iterations + 1):
        samples_0, samples_1 = flow.get_data()
        samples_0 = samples_0.detach().clone()
        samples_1 = samples_1.detach().clone()
        pairs = torch.stack([samples_0, samples_1], dim=1)

        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:args.flow_train_bs]

        batch = pairs[indices]

        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = flow.get_train_tuple(z0=z0, z1=z1)

        pred, _, _ = flow.model(z0, z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(i, np.log(loss.item()))
        if i % 100 == 0:
            torch.save(flow.model.state_dict(), args.flow_file)
            print(f"Model saved to {args.flow_file}")

            samples_0 = flow.noise.sample(2000).to(args.device)
            # samples_0 = torch.sigmoid(torch.randn(2000, args.n_items)).to(
            #     args.device) * 0.1 + 0.45

            if not torch.cuda.is_available():
                draw_plot(flow, z0=samples_0, z1=D.Normal(torch.round(samples_0), 0.01).sample().detach().clone(),
                          N=5, file_name=f'data_flow/flow_init/{i}.pdf')

        loss_curve.append(np.log(loss.item()))

    return flow, loss_curve
