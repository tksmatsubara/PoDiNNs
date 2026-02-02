from main import *


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
    trainer.load_state_dict(torch.load(args.path_tar_original))

    trainer.dataset.data_test = trainer.dataset.data_train
    trainer.u_test = trainer.u_train
    trainer.t_eval_test = trainer.t_eval_train
    trainer.test(stats)

    pickle.dump(stats, open(args.path_pkl, "wb"))

    return trainer, stats


if __name__ == "__main__":
    args = get_args()
    result_dir = f"results/{args.dataset}TRAINEVAL/"
    args.path_tar_original = args.path_tar
    os.makedirs(result_dir) if not os.path.exists(result_dir) else None
    args.path_tar = args.path_tar.replace(args.dataset, args.dataset + "TRAINEVAL")
    args.path_pkl = args.path_pkl.replace(args.dataset, args.dataset + "TRAINEVAL")
    args.path_txt = args.path_txt.replace(args.dataset, args.dataset + "TRAINEVAL")
    args.path_swp = args.path_swp.replace(args.dataset, args.dataset + "TRAINEVAL")
    args.path_dir = args.path_dir.replace(args.dataset, args.dataset + "TRAINEVAL")

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
