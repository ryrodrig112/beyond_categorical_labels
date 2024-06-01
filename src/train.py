from src.utils import Model, Trainer, ClsModel, RgrModel
from src.data import CIFAR10DLGetter, CIFAR10Extended
from src.models import HighDimModel, CategoricalModel, BaselineModel
import torch
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import os
from datetime import datetime
from pathlib import Path
import argparse
import json

def sayhi():
    print("yay")
def load_config_file(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def config_model(config: dict, high_dim_label_set):
    """Unpack dictionary of model hyperparameters
    config: dictionary of model hyperparameters
    high_dim_label_set: the unique set of high dim labels associated w/ each class, needed to init high dim model
    """

    label_type = config["label_type"]
    model_name = config["model_name"]
    latent_size = config.get("latent_size", 64)  # Default value if not in config
    lr = config["lr"]
    optimizer_type = config["optimizer_type"]
    batch_size = config["batch_size"]
    num_classes = config.get("num_classes", 10)  # Default value if not in config
    momentum = config.get("momentum", 0.9)  # Default value if not in config
    weight_decay = config.get("weight_decay", 0.0005)  # Default value if not in config
    lr_decay_schedule = config.get("lr_decay_schedule", None)
    gamma = config.get("lr_decay", 0.1)

    model_load_path = config.get("model_load_path", None)
    model_epoch_nm = config.get("epoch_number", 0)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model based on label type
    if label_type == "high_dim":
        network = HighDimModel(model_name, latent_size, num_classes)
        loss_fn = nn.HuberLoss()
    else:
        # network = CategoricalModel(model_name, num_classes)
        network = BaselineModel(model_name, num_classes)
        print(network.children)
        loss_fn = nn.CrossEntropyLoss()

    # Init optimizer the optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr,
                                    momentum,
                                    weight_decay)

    if lr_decay_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_schedule, gamma)
    else:
        scheduler = None

    model_dir_suffix = f"{model_name}_{label_type}"
    if label_type == "categorical":
        model = ClsModel(network, optimizer, loss_fn, scheduler, model_dir_suffix)
    elif label_type == "high_dim":
        model = RgrModel(network, optimizer, loss_fn,scheduler, high_dim_label_set, model_dir_suffix)

    return model, batch_size

def configure_logging(run_config, train_pct=0.9, trial=1):
    """
    Configures and sets up logging directories for storing model runs and results.
    It creates a unique directory for each run based on the current timestamp, with
    subdirectories for models and results. Returns the paths to these directories.
    """
    logging_root = Path(run_config["logging_root"])
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    train_pct_label = "trainpct" + f"{train_pct}"
    trial_label = "trial" + f"{trial+1}"
    run_name = '_'.join([run_time, train_pct_label, trial_label])
    run_dir = logging_root / run_name
    model_dir = run_dir / "models"
    results_dir = run_dir / "results"

    # Create the directories, ignoring if they already exist
    for directory in [run_dir, model_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)


    print("logging made")

    return results_dir, model_dir


def load_datasets(run_config, model_config, train_pct):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]

    classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    label_map = {idx: label for idx, label in enumerate(classes)}
    high_dim_label_path = run_config['high_dim_label_path']

    cifar_data = CIFAR10DLGetter(train_pct,
                                 run_config['val_pct'],
                                 model_config['batch_size'],
                                 model_config['label_type'],
                                 high_dim_label_path, label_map)

    high_dim_label_set = cifar_data.high_dim_labels

    train_loader = cifar_data.get_trainloader()
    val_loader = cifar_data.get_valloader()
    test_loader = cifar_data.get_testloader()
    return train_loader, val_loader, test_loader, high_dim_label_set

def print_run_info(trainer, model_config, print_rate):
    print(f"Beginning training: {model_config['label_type']} {model_config['model_name']}")
    print()
    print(f"Model parameters: {trainer.model.count_parameters()}")
    print(f"Size of training dataset: {len(trainer.trainloader.dataset)}")
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Batches in trainloader: {len(trainer.trainloader)}")
    print(f"Printing every {print_rate} batches")
    print(f"Logging at {trainer.log_dir}")
    print()


def run_training(model_config_path, run_config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = load_config_file(model_config_path)
    run_config = load_config_file(run_config_path)
    num_trials = run_config.get("num_trials", 0)

    # Init model and data
    train_pcts = run_config["train_pct"]
    val_pct = run_config["val_pct"]
    for train_pct in train_pcts:
        print(f"Training with {100 * train_pct}% of training data")
        for trial in range(num_trials):
            print(f"Trial #{trial + 1}")
            train_loader, val_loader, test_loader, high_dim_label_set = load_datasets(run_config,
                                                                                      model_config,
                                                                                      train_pct)
            model, batch_size = config_model(model_config, high_dim_label_set)
            print_rate = int((len(train_loader)) * run_config["print_pct"])
            results_dir, model_dir = configure_logging(run_config, train_pct, trial)

            trainer = Trainer(model,
                              train_loader,
                              val_loader,
                              results_dir,
                              device=device,
                              model_save_dir=model_dir)
            print_run_info(trainer, model_config, print_rate)
            print(f"Batches in training set: {len(train_loader)}")
            print(f"Batches in validation set: {len(val_loader)}")
            print(f"{len(val_loader.dataset)}")
            # Run training
            print(f"Trainer Device: {trainer.device}")
            # trainer.run_training(run_config["num_epochs"], batch_print_rate=print_rate)
    return model, test_loader

if __name__ == "__main__":
    # Load run and model configuration
    model_config_path = "../config/models/high_dim/vgg11_high_dim.json"
    run_config_path = "../config/data_constrained_highdim_run_params.json"
    model_config = load_config_file(model_config_path)
    run_config = load_config_file(run_config_path)
    num_trials = run_config.get("num_trials", 0)

    # Init model and data
    train_pcts = run_config["train_pct"]
    val_pct = run_config["val_pct"]

    for train_pct in train_pcts:
        print(f"Training with {100 * train_pct}% of training data")
        for trial in range(num_trials):
            print(f"Trial #{trial+1}")
            train_loader, val_loader, test_loader, high_dim_label_set = load_datasets(run_config, model_config, train_pct)
            model, batch_size = config_model(model_config, high_dim_label_set)
            print_rate = int((len(train_loader)) * run_config["print_pct"])
            results_dir, model_dir = configure_logging(run_config, train_pct, trial)
            trainer = Trainer(model, train_loader, val_loader, results_dir, model_save_dir=model_dir)
            if run_config.get("model_load_path"):
                trainer.load_model(run_config["model_load_path"])

            print_run_info(trainer, model_config, print_rate)
            print(f"Batches in training set: {len(train_loader)}")
            print(f"Batches in validation set: {len(val_loader)}")
            print(f"{len(val_loader.dataset)}")
            # Run training
            trainer.run_training(run_config["num_epochs"], batch_print_rate=print_rate)