import torch
import torch.backends.cudnn
import torch.backends.cuda

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.set_default_dtype(torch.float32)
import argparse
import numpy as np
import os
import sys
import importlib
import pickle


def get_args():
    parser = argparse.ArgumentParser(description=None)
    # models
    parser.add_argument("--model", default="podinn", type=str, help="model.")
    parser.add_argument("--act", default="tanh", type=str, help="neural net activation function")
    parser.add_argument("--hidden_dim", default=200, type=int, help="hidden dimensionality of mlp")
    parser.add_argument("--n_layers", default=3, type=int, help="number of layers")
    parser.add_argument("--solver", default="DOPRI45", type=str, help="solver")
    parser.add_argument("--linear_C", action="store_true", help="Force C-type elements (spring or capacitors) to be linear")
    parser.add_argument("--linear_I", action="store_true", help="Force I-type elements (mass or inductors) to be linear")
    parser.add_argument("--set_d", default=None, type=int, help="number of additional damper")
    parser.add_argument("--set_g", default=None, type=int, help="number of additional dual damper")
    parser.add_argument("--prefix", default=None, type=str, help="prefix for saved data")
    # training
    parser.add_argument("--dataset", default="msdFrel", type=str, help="dataset")
    parser.add_argument("--total_steps", default=100000, type=int, help="number of gradient steps")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=100, type=int, help="batch_size")
    parser.add_argument("--max_prediction_step", default=1, type=int, help="number of steps for prediction")
    # display
    parser.add_argument("--log_freq", default=200, type=int, help="number of gradient steps between prints")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--noretry", action="store_true", help="not performing a finished trial")
    args = parser.parse_args()

    result_dir = f"results/{args.dataset}/"
    os.makedirs(result_dir) if not os.path.exists(result_dir) else None
    label = result_dir
    label = label + f"{args.prefix}" if args.prefix else label
    label = label + f"{args.model}"
    if args.set_d is not None or args.set_g is not None:
        label = label + f"+"
        label = label + f"d{args.set_d}" if args.set_d is not None else label
        label = label + f"g{args.set_g}" if args.set_g is not None else label
    label = label + f"+{args.solver}" if args.solver != "DOPRI45" else label
    label = label + f"-seed{args.seed}"
    args.path_tar = f"{label}.tar"
    args.path_pkl = f"{label}.pkl"
    args.path_txt = f"{label}.txt"
    args.path_swp = f"{label}.swp"
    args.path_dir = f"{label}"

    return args


def train(args):
    # initialize
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.dtype = torch.get_default_dtype()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stats = {}

    # load dataset & model
    datasetlib = importlib.import_module(f"dataset.{args.dataset}")
    dataset = datasetlib.Dataset(base_dir="dataset/")
    import model

    trainer = model.get_trainer(dataset, args).to(device=args.device)

    # training
    trainer.train(stats)
    trainer.test(stats)

    pickle.dump(stats, open(args.path_pkl, "wb"))
    torch.save(trainer.state_dict(), args.path_tar)

    dataset.visualize(args.path_dir, stats["u_predicted"])
    return trainer, stats


if __name__ == "__main__":
    args = get_args()

    if args.noretry:
        if os.path.exists(args.path_txt):
            print(args.path_txt)
            print("====== already done:", " ".join(sys.argv), flush=True)
            exit()
        if os.path.exists(args.path_swp):
            print(args.path_swp)
            print("====== already performing:", " ".join(sys.argv), flush=True)
            exit()
        else:
            with open(args.path_swp, "w") as _:
                pass

    print("====== not yet:", " ".join(sys.argv), flush=True)
    trainer, stats = train(args)
    keys = [k for k in stats.keys() if "loss" in k]
    with open(args.path_txt, "w") as of:
        print("#", *keys, sep="\t", file=of)
        for itr in range(args.total_steps):
            print(*[stats[k][itr] if len(stats[k]) > itr else "*" for k in keys], sep="\t", file=of)

    print("====== ended:", " ".join(sys.argv), flush=True)
