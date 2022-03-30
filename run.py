"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os
import math

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

# imported for optimizer initialization
import geoopt as geo

# imported for HPO
import optuna

# for copying args without reference passing
from copy import copy

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WN18RR", choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10"],
    help="Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=4, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=16, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
# custom arguments
parser.add_argument(
    "--curv", default=4, type=float, help="Sets the curvature for models that support it"
)
parser.add_argument(
    "--hpoTrials", default=10, type=int, help="Sets the number of HPO rounds"
)
parser.add_argument(
    "--hpoSampler", default="grid", type=str, help="Selects the sampling method (grid (grid search), rand (random sampler), tpe (Tree Parzen Estimator), etc.)"
)
parser.add_argument(
    "--hyperbolicCurv", default=4, type=float, help="DONT USE (for internal use)"
)
parser.add_argument(
    "--sphericalCurv", default=4, type=float, help="DONT SET (for internal use)"
)
parser.add_argument(
    "--non_euclidean_ratio", default=None, type=float, help="DONT SET (for internal use)"
)
parser.add_argument(
    "--hyperbolic_ratio", default=None, type=float, help="DONT SET (for internal use)"
)


def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create empty model
    model = None

    # get optimizer
    optimizer = None

    counter = 0
    best_mrr = 0
    best_epoch = None
    logging.info("\t Start training")
    
    step = 0
    valid_mrr = 0

    # default args
    defArgs = copy(args)

    def train_model():
        nonlocal step
        nonlocal train_examples
        nonlocal model
        nonlocal optimizer

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))
        
    def validate_model():
        nonlocal step
        nonlocal model
        nonlocal valid_examples
        nonlocal args
        nonlocal filters
        nonlocal valid_mrr
        nonlocal model
        nonlocal optimizer

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]

    def objective(trial):
        nonlocal best_mrr
        nonlocal step
        nonlocal save_dir
        nonlocal model
        nonlocal valid_mrr
        nonlocal counter
        nonlocal best_epoch
        nonlocal model
        nonlocal optimizer
        
        valid_mrr = 0
        best_mrr = 0

        # set params
        # Exponential sampling
        args.rank = int(round(pow(2, trial.suggest_float("args.rank", math.log(16, 2), math.log(defArgs.rank, 2))), 0))
        args.curv = round(trial.suggest_float("args.curv", 0, defArgs.curv), 4)
        # Exponential sampling
        args.learning_rate = round(pow(2, trial.suggest_float("args.learning_rate", math.log(0.0001, 2), math.log(defArgs.learning_rate, 2))), 10)
        #non_euclidean_optimizer = trial.suggest_categorical("non_euclidean_optimizer", ["RiemannianAdam", "RiemannianLineSearch", 
        #"RiemannianSGD"])
        non_euclidean_optimizer = trial.suggest_categorical("non_euclidean_optimizer", ["RiemannianAdam", "RiemannianLineSearch", 
        "RiemannianSGD"])
        
        if defArgs.model == "mixed":
            args.hyperbolicCurv = round(trial.suggest_float("args.hyperbolicCurv", 0, defArgs.curv), 4)
            args.sphericalCurv = round(trial.suggest_float("args.sphericalCurv", 0, defArgs.curv), 4)
            args.non_euclidean_ratio = round(trial.suggest_float("args.non_euclidean_ratio", 0, 1), 4)
            args.hyperbolic_ratio = round(trial.suggest_float("args.hyperbolic_ratio", 0, 1), 4)
        
        # create model
        model = getattr(models, args.model)(args)
        total = count_params(model)
        logging.info("Total number of parameters {}".format(total))
        # replace cuda with cpu in order to use the cpu
        device = "cuda"
        model.to(device)

        # get optimizer
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
        # Set Riemannian Optimizer
        if args.model == "hyperbolic" or args.model == "spheric" or args.model == "mixed" or args.model == "euclidean":
            if str(non_euclidean_optimizer) == "RiemannianLineSearch":
                optim_method.optimizer = getattr(geo.optim, str(non_euclidean_optimizer))(model.parameters())
            else:
                optim_method.optimizer = getattr(geo.optim, str(non_euclidean_optimizer))(model.parameters(), lr=args.learning_rate)
        optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                                bool(args.double_neg))

        for step in range(args.max_epochs):

            # Train step
            train_model()
            
            # Valid step
            validate_model()
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                # replace model.cuda() with model.cpu() in order to use the cpu
                model.cuda()
            else:
                counter += 1
                if counter == args.patience and (step + 1) % args.valid == 0:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()

        return best_mrr

    # Select sampler
    if defArgs.hpoSampler == "grid":
        search_space = {}
        if defArgs.model == "mixed":
            search_space = {"args.rank": [math.log(16, 2), math.log(defArgs.rank,2)], "args.curv": [0, defArgs.curv], 
            "args.learning_rate": [math.log(0.0001, 2), math.log(defArgs.learning_rate, 2)], "non_euclidean_optimizer": ["RiemannianAdam", "RiemannianLineSearch", "RiemannianSGD"],
            "args.hyperbolicCurv": [0, defArgs.hyperbolicCurv], "args.sphericalCurv": [0, defArgs.sphericalCurv], 
            "args.non_euclidean_ratio": [0, 1], "args.hyperbolic_ratio": [0, 1]}
            #search_space = {"args.hyperbolicCurv": [0, defArgs.hyperbolicCurv], "args.sphericalCurv": [0, defArgs.sphericalCurv], 
            #"args.non_euclidean_ratio": [0, 1], "args.hyperbolic_ratio": [0, 1]}
            #search_space = {"args.hyperbolicCurv": [0, defArgs.hyperbolicCurv], "args.non_euclidean_ratio": [0, 1], "args.hyperbolic_ratio": [0, 1]}
        else:
            search_space = {"args.rank": [math.log(16, 2), math.log(defArgs.rank, 2)], "args.curv": [0, defArgs.curv], 
            "args.learning_rate": [math.log(0.0001, 2), math.log(defArgs.learning_rate, 2)], "non_euclidean_optimizer": ["RiemannianAdam", "RiemannianLineSearch", "RiemannianSGD"]}
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
    elif defArgs.hpoSampler == "rand":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    elif defArgs.hpoSampler == "tpe":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        
    # execute HPO
    study.optimize(objective, n_trials=args.hpoTrials, gc_after_trial=True)

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        # disabled since HPO interveres
        pass
        #logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        #model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    # replace .cuda() with .cpu() to use the cpu
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())
