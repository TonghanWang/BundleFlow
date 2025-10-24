import os
import yaml
import torch
import shutil
import argparse
import numpy as np
from datetime import datetime
import torch.distributions as D
import matplotlib.pyplot as plt

from menu import Menu
from utils import draw_plot
from flow import RectifiedFlow
from flow import train as train_flow_network
from menu import train as train_menu_network


def parse_arguments():


    parser = argparse.ArgumentParser(
        description="Solve 1-bidder combinatorial auction with flow matching."
    )

    parser.add_argument('--valuation', type=str, default='additive',
                        choices=['combinatorial', 'additive'],
                        help='Valuation type (default: additive).')

    # Parse only known arguments here (so we know which config to load)
    args, _ = parser.parse_known_args()

    config_file = 'config/combinatorial.yaml' if args.valuation == 'combinatorial' else 'config/additive.yaml'

    # Load the YAML config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Based on --valuation, get the corresponding default dict
    valuation_defaults = config

    # Auction
    parser.add_argument('--n_items', type=int)

    # Menu representation
    parser.add_argument('--J', type=int)
    parser.add_argument('--menu_support_size', type=int)

    # Training the flow network
    parser.add_argument('--n_flow_samples', type=int)
    parser.add_argument('--flow_train_bs', type=int)
    parser.add_argument('--flow_train_iterations', type=int)
    parser.add_argument('--flow_nn_hidden', type=int)

    # Train the menu network
    parser.add_argument('--menu_train_bs', type=int)
    parser.add_argument('--menu_train_iterations', type=int)
    parser.add_argument('--menu_print_iterations', type=int)
    parser.add_argument('--menu_eval_iterations', type=int)
    parser.add_argument('--train_ratio', type=float)
    parser.add_argument('--menu_lr', type=float)
    parser.add_argument('--min_menu_lr', type=float)
    parser.add_argument('--menu_lr_decay', type=float)
    parser.add_argument('--init_softmax', type=float)
    parser.add_argument('--test_bs', type=int)
    parser.add_argument('--n_test_batches', type=int)
    parser.add_argument('--value_file', type=str)

    # Now override argparse defaults with the selected valuation defaults
    parser.set_defaults(**valuation_defaults)
    args = parser.parse_args()

    # Use a larger flow network and menu for large problems
    if args.n_items > 100:
        args.flow_nn_hidden = 256
    # if args.n_items >= 100:
    #     args.J = 5000

    return args


if __name__ == "__main__":
    args = parse_arguments()
    args.device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    current_time = datetime.now()
    runner_id = current_time.strftime("%Y%m%d_%H%M%S_%f")
    args.runner_id = runner_id
    args.log_dir = os.path.join('logs', args.runner_id)
    args.flow_file = os.path.join('logs', f'flow_models/flow_{args.n_items}.pth')

    # Flow
    if not os.path.exists(args.flow_file):
        flow_model = RectifiedFlow(args)
        flow_model, loss_curve = train_flow_network(flow_model, args)
        torch.save(flow_model.model.state_dict(), args.flow_file)
        print(f"Model saved to {args.flow_file}")

        plt.plot(np.linspace(0, args.flow_train_iterations, args.flow_train_iterations + 1),
                 loss_curve[:(args.flow_train_iterations + 1)])
        plt.title('Training Loss Curve')
        plt.show()
    else:
        # loaded_model.eval()  # Set to evaluation mode if necessary
        flow_model = RectifiedFlow(args)
        flow_model.load(args.flow_file)

    samples_0 = flow_model.noise.sample(128).to(device)
    if not torch.cuda.is_available():
        draw_plot(flow_model, z0=samples_0, z1=D.Normal(torch.round(samples_0), 0.01).sample().detach().clone(), N=5)

    # Menu
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.mkdir(args.log_dir)

    for param in flow_model.model.parameters():
        param.requires_grad = False

    menu = Menu(flow_model, args)

    # samples_0 = menu.support[0]
    # if not torch.cuda.is_available():
    #     draw_plot(flow_model, z0=menu.support[0], z1=D.Normal(torch.round(samples_0), 0.01).sample().detach().clone(), N=5)

    train_menu_network(menu, args)
