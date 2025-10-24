# This version uses optimized memory usage
# The probability of each possible value is assumed to come from a Mixture of Gaussian distribution

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset


class Menu(nn.Module):
    def __init__(self, rectified_flow, args: argparse.Namespace):
        super(Menu, self).__init__()
        self.rectified_flow = rectified_flow

        # Allocation
        self.n_components = args.menu_support_size
        # self.initial_means = nn.Parameter(
        #     torch.rand(args.J, self.n_components, args.n_items) / 100 + 0.5)  # [J, support_size, n_items]
        self.initial_means = nn.Parameter(
            torch.randn(args.J, self.n_components, args.n_items))  # [J, support_size, n_items]
        self.initial_weights = nn.Parameter(
            torch.randn(args.J, self.n_components) / self.n_components)  # [J, support_size]

        self.support_size = args.menu_support_size
        self.initial_std = torch.exp(
            (-torch.rand(args.J, self.n_components, args.n_items) * 30 - 4).to(args.device))  # [J, support_size, n_items]

        # Prices
        self.prices = nn.Parameter(torch.rand(args.J))  # [J]
        self.to(args.device)

        self.args = args

        self.softmax_lambda = args.init_softmax

        self.zeros = torch.zeros(args.menu_train_bs, 1).to(args.device)
        self.price_zeros = torch.zeros(1).to(args.device)

    def item_utility(self, value_function):
        # Sample component indices based on the weights
        # t0 = time.time()

        # Construct the initial distribution
        log_weights = torch.log(torch.softmax(self.softmax_lambda * self.initial_weights,
                                              dim=1) + 1e-20)  # Account for the mixture weights in the log probabilities. The log weights corresponding to each component index
        # log_weights: [J, support_size]
        initial_means = self.args.flow_init_dist_start + self.args.flow_init_dist_range * torch.sigmoid(self.initial_means * 10)

        normal = D.Normal(initial_means, self.initial_std)
        z0 = normal.rsample()
        # z0: [J, support_size, n_items]
        log_probs = normal.log_prob(z0).sum(dim=-1)  # Sum log probs across dimensions for multivariate normals
        # log_probs: [J, support_size]
        log_probs += log_weights
        # log_probs: [J, support_size]

        # Transport the support of the initial distribution to the distribution over bundles
        trajs, Q, int_sigma = self.rectified_flow.sample_ode_grad(z0=z0.view(-1, self.args.n_items), N=5)
        trajs = trajs[-1].view(self.args.J, self.support_size, self.args.n_items)
        # trajs: [J, support_size, n_items]

        # trace_Q = torch.diagonal(Q, dim1=-2, dim2=-1).sum(dim=-1).view(self.args.J, self.support_size)
        trace_Q = Q.sum(dim=-1).view(self.args.J, self.support_size)
        # trace_Q: [J, support_size]

        int_sigma = int_sigma.view(self.args.J, self.support_size)
        # int_sigma: [J, support_size]

        ode_prob = trace_Q * int_sigma

        bundles = torch.where(trajs < 0.5, torch.zeros_like(trajs), torch.ones_like(trajs))
        # bundles = torch.sigmoid((trajs - 0.5) * 1000)
        # bundles: [J, support_size, n_items]

        log_probs_n = log_probs - ode_prob
        # log_probs_p = log_probs + ode_prob
        # log_probs_n = log_probs_n * self.args.n_items
        normalized_probs_n = torch.exp(log_probs_n - torch.logsumexp(log_probs_n, dim=-1, keepdim=True))
        # normalized_probs_p = torch.exp(log_probs_p - torch.logsumexp(log_probs_p, dim=-1, keepdim=True))
        # log_probs: [J, support_size]

        if self.args.valuation == 'additive':
            # Compute weighted_bundle_sum: [J, n_items]
            weighted_bundle_sum = (normalized_probs_n.unsqueeze(2) * bundles).sum(1)  # [J, n_items]
            # Compute utility: [bs, J]
            utility_n = torch.matmul(value_function, weighted_bundle_sum.T)  # [bs, J]
            # Subtract prices
            utility_n -= self.prices.unsqueeze(0)
        elif self.args.valuation == 'combinatorial':
            bid_bundles = value_function[:, :, :-1]  # [bs, K, n_items]
            bundles = bundles.view(-1, self.args.n_items)  # [A, n_items]
            values = value_function[:, :, -1]  # [bs, K]
            # # Unified
            # included_mask = (bid_bundles <= bundles.unsqueeze(1).unsqueeze(1)).all(dim=-1)  # [A, bs, K]
            # sub_bundle_values = included_mask * (values.unsqueeze(0))  # [A, bs, K]
            # bidder_values, _ = sub_bundle_values.max(dim=-1)  # [A, bs]
            # # Chunked
            chunk_size = 10000  # choose a size that fits your GPU/CPU memory better
            A_total = bundles.shape[0]  # A
            all_bidder_values = []
            for start_idx in range(0, A_total, chunk_size):
                end_idx = min(start_idx + chunk_size, A_total)
                # slices: [chunk_size, bs, n_items] for bid_bundles, etc.
                bundles_chunk = bundles[start_idx: end_idx]  # [chunk_size, n_items]
                # included_mask_chunk: [chunk_size, bs, K]
                # included_mask_chunk = (bid_bundles <= bundles_chunk.unsqueeze(1).unsqueeze(1)).all(dim=-1)
                bids_2d = bid_bundles.view(-1, self.args.n_items)  # => [bs*K, n_items]
                chunk_inverted = 1 - bundles_chunk  # => [chunk_size, n_items]
                chunk_inverted_T = chunk_inverted.t()  # => [n_items, chunk_size]
                scores = bids_2d.float() @ chunk_inverted_T.float()
                included_2d = (scores == 0)  # [bs*K, chunk_size]
                included_2d = included_2d.t()  # [chunk_size, bs*K]
                included_mask_chunk = included_2d.view(bundles_chunk.shape[0], value_function.shape[0], value_function.shape[1])
                # sub_bundle_values_chunk: [chunk_size, bs, K]
                sub_bundle_values_chunk = included_mask_chunk * (values.unsqueeze(0))
                # reduce along K to get bidder values: [chunk_size, bs]
                bidder_values_chunk, _ = sub_bundle_values_chunk.max(dim=-1)
                # accumulate
                all_bidder_values.append(bidder_values_chunk)

            # now cat them back along dim=0: [A, bs]
            bidder_values = torch.cat(all_bidder_values, dim=0)
            bidder_values = torch.transpose(bidder_values, 0, 1)  # [bs, A]
            bidder_values = bidder_values.view(-1, self.args.J, self.support_size)  # [bs, J, support_size]
            # bidder_values_p = (bidder_values * normalized_probs_p.unsqueeze(0)).sum(-1)  # [bs, J]
            bidder_values_n = (bidder_values * normalized_probs_n.unsqueeze(0)).sum(-1)  # [bs, J]
            # utility_p = bidder_values_p - self.prices.unsqueeze(0)
            utility_n = bidder_values_n - self.prices.unsqueeze(0)
        else:
            raise NotImplementedError

        return utility_n

    # def v(self, bundle, value):
    #     # bundle: [x, n_items]
    #     # value: [bs, n_items]
    #     result = torch.matmul(value, bundle.T)  # [bs, x]
    #     return result

    def softmax_revenue(self, values):
        uj = self.item_utility(values)  # (bs, J)
        uj_ir = torch.cat([uj, self.zeros], dim=1)  # (bs, J+1)

        rev = torch.softmax(uj_ir * self.softmax_lambda, dim=1)[:, :-1]  # [bs, J]
        rev = torch.matmul(rev, self.prices)

        # rev = torch.softmax(uj_ir * self.softmax_lambda, dim=1)  # [bs, J + 1]
        # rev = torch.matmul(rev, torch.cat([self.prices, self.price_zeros], dim=0))

        # rev = torch.softmax(uj_ir * self.softmax_lambda, dim=1)  # [bs, J]
        # rev = rev[:, :-1] - rev[:, -1].unsqueeze(1)
        # rev = torch.matmul(rev, self.prices)

        return rev.mean()

    @torch.no_grad()
    def revenue(self, values):
        uj = self.item_utility(values)  # [bs, J]
        payments = self.prices[torch.argmax(uj, dim=1)]  # [bs]
        payments[uj.max(dim=1)[0] < 0.] = 0.
        return payments.mean()

    @torch.no_grad()
    def get_batch(self, bs, value_data=None):
        if self.args.valuation == 'additive':
            return torch.rand((bs, self.args.n_items)).to(self.args.device)
        elif self.args.valuation == 'combinatorial':
            random_indices = torch.randint(
                low=0,
                high=value_data.shape[0],
                size=(bs,)
            )
            return value_data[random_indices]


def train(menu, args):
    optimizer = torch.optim.Adam(menu.parameters(), lr=args.menu_lr)
    data_file = None

    std1 = 0
    std2 = False

    if args.valuation == 'combinatorial':
        data_file = torch.load(args.value_file).to(args.device)

        # Split train and test set
        n_total_samples = data_file.shape[0]
        n_training_samples = int(n_total_samples * args.train_ratio)
        n_test_samples = n_total_samples - n_training_samples
        test_data_file = data_file[n_training_samples:]
        data_file = data_file[:n_training_samples]

        print('Data set size: ', n_training_samples, n_test_samples)

        dataset = TensorDataset(data_file)
        data_loader = DataLoader(
            dataset,
            batch_size=args.menu_train_bs,
            shuffle=True,
            drop_last=True
        )

        # Create an iterator over the DataLoader
        data_iter = iter(data_loader)
    else:
        data_loader = None
        data_iter = None

    # data_file = None
    # if args.valuation == 'combinatorial':
    #     data_file = torch.load(args.value_file)

    # Initialize the dictionary to store training information
    training_info = {
        "latest_revenue": None,
        "latest_iteration": None,
        "latest_training_samples": None,
        "latest_wall_time": None,
        "best_revenue": None,
        "best_iteration": None,
        "best_training_samples": None,
        "best_wall_time": None,
        "trajectories": [],
        "eval_trajectories": []
    }

    # Variables to track training time and best revenue
    start_time = time.time()
    best_revenue = float('-inf')
    total_training_samples = 0

    scaler = torch.cuda.amp.GradScaler()
    last_val_rev = None

    for i in range(1, args.menu_train_iterations + 1):
        if data_loader is not None:
            try:
                batch_data = next(data_iter)
            except StopIteration:
                # If we've exhausted the DataLoader, re-create (this also reshuffles)
                data_iter = iter(data_loader)
                batch_data = next(data_iter)

            values = batch_data[0]
        else:
            # For additive valuation
            values = menu.get_batch(args.menu_train_bs)

        optimizer.zero_grad()
        # loss = -menu.softmax_revenue(values)
        # loss.backward()
        # optimizer.step()
        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         record_shapes=True,
        #         profile_memory=True
        # ) as p:
        with torch.cuda.amp.autocast():
            loss = -menu.softmax_revenue(values)
        scaler.scale(loss).backward()

        # print(p.key_averages().table(sort_by="cuda_time_total"))
        scaler.step(optimizer)
        scaler.update()
        total_training_samples += args.menu_train_bs

        if i % args.menu_print_iterations == 0 or i == args.menu_train_iterations:
            elapsed_time = time.time() - start_time
            training_info["trajectories"].append({
                "iteration": i,
                "loss": loss.item(),
                "training_time": elapsed_time,
                "training_samples": total_training_samples
            })
            print("[Iter]: %d, [Loss]: %.6f, [Softmax Lambda]: %.6f" % (i, loss.item(), menu.softmax_lambda))

            weights = torch.softmax(menu.softmax_lambda * menu.initial_weights, dim=1)
            show_initial_means = menu.args.flow_init_dist_start + menu.args.flow_init_dist_range * torch.sigmoid(
                menu.initial_means * 10).mean(-1)

            std = (weights * show_initial_means).sum(-1).mean().item()

            print((weights * show_initial_means).max(-1)[0].mean().item(), std)
            print(show_initial_means.mean().item())
            print(menu.prices.mean().item(), menu.prices.max().item(), menu.prices.min().item())


            # if menu.softmax_lambda < 10:
            #     menu.softmax_lambda *= 1.1
            # else:
            menu.softmax_lambda = min(0.2, menu.softmax_lambda * args.factor_softmax)

            if menu.softmax_lambda >= 0.2:
                args.factor_softmax = 1.05

        if i % (args.menu_print_iterations * 2) == 0 and std > 0.5 and args.factor_softmax > 1.01:
            if std < std1:
                menu.softmax_lambda = 0.05
                args.factor_softmax = 1.01
            std1 = std

        if i % args.menu_eval_iterations == 0 or i == args.menu_train_iterations:
            torch.save(menu, f'{args.log_dir}/menu_{i}.pt')

            test_bs = args.test_bs

            if args.valuation == 'combinatorial':
                num_val_batches = n_test_samples // test_bs
                with torch.no_grad():
                    val_rev = 0.0
                    for j in range(num_val_batches):
                        values = test_data_file[j * test_bs: (j + 1) * test_bs]
                        val_rev += menu.revenue(values).detach()
                    avg_val_rev = val_rev.item() / num_val_batches
            else:
                num_val_batches = args.n_test_batches
                with torch.no_grad():
                    val_rev = 0.0
                    for j in range(num_val_batches):
                        values = menu.get_batch(test_bs)
                        val_rev += menu.revenue(values).detach()
                    avg_val_rev = val_rev.item() / num_val_batches

            # Update training information
            elapsed_time = time.time() - start_time
            training_info["latest_revenue"] = avg_val_rev
            training_info["latest_iteration"] = i
            training_info["latest_training_samples"] = total_training_samples
            training_info["latest_wall_time"] = elapsed_time

            if avg_val_rev > best_revenue:
                best_revenue = avg_val_rev
                training_info["best_revenue"] = best_revenue
                training_info["best_iteration"] = i
                training_info["best_training_samples"] = total_training_samples
                training_info["best_wall_time"] = elapsed_time

            training_info["eval_trajectories"].append({
                "iteration": i,
                "avg_val_revenue": avg_val_rev,
                "wall_time": elapsed_time,
                "training_samples": total_training_samples
            })

            print("\tTest Revenue", avg_val_rev)

            # Save training information to a file
            torch.save(training_info, f'{args.log_dir}/Diff_{args.valuation}_{args.value_file.split("/")[-1]}')

            if last_val_rev is not None and avg_val_rev < last_val_rev:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * args.menu_lr_decay, args.min_menu_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"\tValidation revenue decreased: LR adjusted from {current_lr:.6g} to {new_lr:.6g}")

                # Update last_val_rev to the current validation result
            last_val_rev = avg_val_rev


def test(menu, args):
    data_file = None
    menu.args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    menu = torch.load('data/menu_3000.pt', map_location=menu.args.device)
    menu.args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.valuation == 'combinatorial':
        data_file = torch.load(args.value_file).to(args.device)

        # Split train and test set
        n_total_samples = data_file.shape[0]
        n_training_samples = int(n_total_samples * 0.95)
        n_test_samples = n_total_samples - n_training_samples
        test_data_file = data_file[n_training_samples:]
        data_file = data_file[:n_training_samples]

        print('Data set size: ', n_training_samples, n_test_samples)

        dataset = TensorDataset(data_file)
        data_loader = DataLoader(
            dataset,
            batch_size=args.menu_train_bs,
            shuffle=True,
            drop_last=True
        )

        # Create an iterator over the DataLoader
        data_iter = iter(data_loader)
    else:
        data_loader = None
        data_iter = None

    if args.valuation == 'combinatorial':
        test_bs = 4
        num_val_batches = n_test_samples // test_bs
        with torch.no_grad():
            val_rev = 0.0
            for j in range(num_val_batches):
                values = test_data_file[j * test_bs: (j + 1) * test_bs]
                val_rev += menu.softmax_revenue(values).detach()
            avg_val_rev = val_rev.item() / num_val_batches
    else:
        test_bs = 10
        num_val_batches = 500
        with torch.no_grad():
            val_rev = 0.0
            for j in range(num_val_batches):
                values = menu.get_batch(test_bs)
                val_rev += menu.revenue(values).detach()
            avg_val_rev = val_rev.item() / num_val_batches

    print("\tTest Revenue", avg_val_rev)
