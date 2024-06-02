import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
from pathlib import Path
from src.data import CIFAR10Extended

class Model:
    def __init__(self, device, network: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 scheduler: torch.optim.lr_scheduler,
                 model_dir_suffix: str): # eventually this will get changed to load hyperparms
        self.device = device
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.model_dir_suffix = model_dir_suffix

    def forward(self, x):
        x.to(self.device)
        self.network.to(self.device)
        return self.network(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def train_one_epoch(self, trainloader, device, batch_print_rate=0):
        ...

class RgrModel(Model):
    def __init__(self, device, network: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 scheduler: torch.optim.lr_scheduler,
                 high_dim_label_set,
                 model_dir_suffix: str,
                 ):
        super().__init__(device, network, optimizer, loss_fn, scheduler, model_dir_suffix
                 )
        self.high_dim_label_set = torch.tensor(high_dim_label_set)

    def calc_loss_across_classes(self, yhat_regr):
        """Given a predicted output, and a tensor containing the high dim labels for all classes, find the
        high dimensional label the prediction is closest to prediction"""
        losses = torch.zeros(yhat_regr.size(0), self.high_dim_label_set.size(0)).to(self.device)
        for i, y_hat in enumerate(yhat_regr):
            for j, label in enumerate(self.high_dim_label_set):
                loss = self.loss_fn(y_hat.to(self.device), label.to(self.device))
                losses[i, j] = loss
        return losses

    def classify_predictions(self, yhat_regr):
        losses = self.calc_loss_across_classes(yhat_regr)
        yhat_cls = torch.argmin(losses, dim=1)
        return yhat_cls

    def calc_cls_accuracy(self, yhat_cls: torch.Tensor, cat_label: torch.Tensor):
        check = yhat_cls == cat_label
        num_accurate = sum(check).item()
        num_inaccurate = yhat_cls.size(0) - num_accurate
        return num_accurate, num_inaccurate

    def train_one_epoch(self, trainloader,
                        batch_print_rate=0,
                        summary_writer=None,
                        epoch=0,
                        writer=None):
        n_batches = len(trainloader)
        n_datapoints = len(trainloader.dataset)
        epoch_loss = 0.0
        total_accurate = 0.0

        display_loss = 0.0
        display_num_accurate = 0.0
        display_total = 0.00

        self.network.train()
        for i, data in enumerate(trainloader, 0):
            self.optimizer.zero_grad() # zero the parameter gradients
            # get the inputs; data is a list of [inputs, high_dim_label, categorical_label]
            inputs, high_dim_labels, cat_labels = data
            inputs, high_dim_labels, cat_labels = inputs.to(self.device), \
                                                  high_dim_labels.to(self.device), \
                                                  cat_labels.to(self.device)
            batch_size = len(inputs)
            yhat_rgr = self.forward(inputs) # forward pass
            yhat_cls = self.classify_predictions(yhat_rgr) # classify into predicted classes

            #calculate loss & accuracy for batch, track for epoch
            loss = self.loss_fn(yhat_rgr, high_dim_labels)
            num_accurate, _ = self.calc_cls_accuracy(yhat_cls, cat_labels)

            epoch_loss += loss.item()
            total_accurate += num_accurate

            display_loss += loss.item()
            display_num_accurate += num_accurate
            display_total += batch_size

            #optimize
            loss.backward()
            self.optimizer.step()
            if (batch_print_rate > 0):
                if (i % batch_print_rate == batch_print_rate - 1):
                    step = ((i+1)/batch_print_rate) + epoch
                    display_accuracy =  display_num_accurate/ display_total
                    print(f" Batch: {i+1:5d}  of {n_batches:5d}, {(100* (i+1))/n_batches:0.0f}% Complete")
                    print(f" Avg Loss: {display_loss/display_total:.2}, "
                          f"Accuracy: {100*display_accuracy:.2f}%")
                    display_loss = 0
                    display_num_accurate = 0
                    display_total = 0
                    if writer:
                        writer.update_metric("Loss", display_loss, "Training", step)
                        writer.update_metric("Accuracy", display_accuracy, "Training", step)
        self.scheduler.step()
        average_tr_loss = epoch_loss / n_datapoints
        tr_accuracy = total_accurate / n_datapoints
        return average_tr_loss, tr_accuracy

    def eval_performance(self, dataloader: DataLoader):
        print("Evaluating")
        self.network.eval()
        total_accurate = 0
        total_loss = 0
        n = len(dataloader.dataset)
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, high_dim_labels, cat_labels = data
                inputs, high_dim_labels, cat_labels = inputs.to(self.device), \
                                                      high_dim_labels.to(self.device), \
                                                      cat_labels.to(self.device)
                yhat_rgr = self.forward(inputs)
                batch_loss = self.loss_fn(yhat_rgr, high_dim_labels)
                total_loss += batch_loss
                yhat_cls = self.classify_predictions(yhat_rgr)
                num_accurate, _ = self.calc_cls_accuracy(yhat_cls, cat_labels)
                total_accurate += num_accurate
        accuracy = total_accurate / n
        avg_loss = total_loss / n
        return avg_loss, accuracy

class ClsModel(Model):
    def __init__(self, device, network: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: nn.Module,
                 scheduler: torch.optim.lr_scheduler,
                 model_dir_suffix: str
                 ):
        super().__init__(device, network, optimizer, loss_fn, scheduler, model_dir_suffix
                 )

    def calc_cls_accuracy(self, yhat_cls: torch.Tensor, cat_label: torch.Tensor):
        check = yhat_cls == cat_label
        num_accurate = sum(check).item()
        num_inaccurate = yhat_cls.size(0) - num_accurate
        return num_accurate, num_inaccurate

    def train_one_epoch(self, trainloader, batch_print_rate=0, summary_writer=None, epoch=0, writer=None):
        n_batches = len(trainloader)
        n_datapoints = len(trainloader.dataset)
        epoch_loss = 0.0
        total_accurate = 0.0

        display_loss = 0.0
        display_num_accurate = 0.0
        display_total = 0.00

        self.network.train()
        for i, data in enumerate(trainloader, 0):
            self.optimizer.zero_grad() # zero the parameter gradients
            # get the inputs; data is a list of [inputs, high_dim_label, categorical_label]
            inputs, labels = data
            inputs, labels= inputs.to(self.device), labels.to(self.device)
            batch_size = len(inputs)
            logits = self.forward(inputs) # forward pass
            preds = torch.argmax(logits, dim=1) # classify into predicted classes

            #calculate loss & accuracy for batch, track for epoch
            loss = self.loss_fn(logits, labels)
            num_accurate, _ = self.calc_cls_accuracy(preds, labels)

            epoch_loss += loss.item()
            total_accurate += num_accurate

            display_loss += loss.item()
            display_num_accurate += num_accurate
            display_total += batch_size

            #optimize
            loss.backward()
            self.optimizer.step()

            if (batch_print_rate > 0) & (i % batch_print_rate == batch_print_rate - 1):
                display_accuracy = display_num_accurate/ display_total
                print(f" Batch: {i+1:5d}  of {n_batches:5d}, {(100* (i+1))/n_batches:0.0f}% Complete")
                print(f" Avg Loss: {display_loss/batch_print_rate:.2}, "
                      f"Accuracy: {100*display_accuracy:.2f}%")
                display_loss = 0
                display_num_accurate = 0
                display_total = 0

        # self.scheduler.step()
        average_tr_loss = epoch_loss / n_batches
        tr_accuracy = total_accurate / n_datapoints

        print(f"Avg Epoch Loss: {average_tr_loss:0.2f}")
        print(f"Avg Epoch Accuracy: {100*tr_accuracy:0.2f}%")
        return average_tr_loss, tr_accuracy

    def eval_performance(self, dataloader: DataLoader):
        total_accurate = 0
        total_loss = 0
        n_batches = len(dataloader)
        n_datapoints = len(dataloader.dataset)
        self.network.eval()
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, high_dim_label, categorical_label]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = len(inputs)
            logits = self.forward(inputs) # forward pass
            preds = torch.argmax(logits, dim=1) # classify into predicted classes

            #calculate loss & accuracy for batch, track for epoch
            loss = self.loss_fn(logits, labels)
            num_accurate, _ = self.calc_cls_accuracy(preds, labels)

            total_loss += loss.item()
            total_accurate += num_accurate
        accuracy = total_accurate / n_datapoints
        avg_loss = total_loss / n_batches
        return avg_loss, accuracy


class Trainer:
    """Given any model with a train_one_epoch function, trains or evals model, strong results and plots"""
    def __init__(self,
                 model: Model,
                 trainloader: DataLoader,
                 valloader:DataLoader,
                 log_prefix: str,
                 model_save_dir='models'):
        if not hasattr(model, 'train_one_epoch'):
            raise ValueError("The provided model does not have a 'train_one_epoch' method.")
        elif not callable(getattr(model, 'train_one_epoch')):
            raise ValueError("The provided model does not have a 'train_one_epoch' method.")
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.model_save_dir = model_save_dir
        self.log_dir = f"{log_prefix}_{self.model.model_dir_suffix}"
        self.writer = SummaryWriter(self.log_dir)

    def run_training(self, epochs, batch_print_rate=0, validation=True):
        print(f"batch print rate from run_training: {batch_print_rate}")
        for epoch in range(epochs):
            print("-----------")
            print(f"Beginning epoch {epoch + 1}")
            train_loss, train_accuracy = self.model.train_one_epoch(self.trainloader,
                                                                    batch_print_rate=batch_print_rate)
            ...
            if validation:
                val_loss, val_accuracy = self.model.eval_performance(self.valloader)
                self.writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'Train': train_accuracy, 'Validation': val_accuracy}, epoch)
                print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                print(f"Train Accuracy: {100 * train_accuracy:.2f}%, Validation Accuracy: {100 * val_accuracy:.2f}%")
            self.save_model(epoch)
        print("Training Complete")

    def save_model(self, epoch):
        file_nm = f"epoch{epoch}.pt"
        model_path = Path(self.model_save_dir) / file_nm
        torch.save(self.model.network.state_dict(), model_path)

    def load_model(self, path):
        self.model.network.load_state_dict(torch.load(path))


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    print(ea.Tags()["scalars"])
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


if __name__ == "__main__":
    dicts = parse_tensorboard("../HuberLoss()/Loss_Train", ["Loss"])
    print(dicts)





