#  Copyright (c) 2021. Lucas G. S. Jeub
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
"""TODO: module docstring for embedding/train.py."""

import torch

class EarlyStopping:
    """
    Context manager for early stopping
    """

    def __init__(self, patience: int, delta: float = 0):
        """
        Initialise early stopping context manager

        Args:
            patience: wait ``patience`` number of epochs without loss improvement before stopping
            delta: minimum improvement to consider significant (default: 0)
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.count = 0
        self._file = TemporaryFile()

    def __enter__(self):
        self.best_loss = float("inf")
        self.count = 0
        if self._file.closed:
            self._file = TemporaryFile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

    def _save_model(self, model: torch.nn.Module):
        self._file.seek(0)
        torch.save(model.state_dict(), self._file)

    def _load_model(self, model: torch.nn.Module):
        self._file.seek(0)
        model.load_state_dict(torch.load(self._file))

    def __call__(self, loss: float, model: torch.nn.Module) -> bool:
        """
        check stopping criterion and save or restore model state as appropriate

        Args:
            loss: loss value for stopping
            model: [description]

        Returns:
            ``True`` if training should be stopped, ``False`` otherwise
        """
        loss = float(
            loss
        )  # make sure no tensors used here to avoid propagating gradients
        if loss >= self.best_loss - self.delta:
            self.count += 1
        else:
            self.count = 0

        if loss < self.best_loss:
            self.best_loss = loss
            self._save_model(model)
        if self.count > self.patience:
            self._load_model(model)
            return True

        return False



def lr_grid_search(
    data,
    model,
    loss_fun,
    validation_loss_fun,
    lr_grid=(0.1, 0.01, 0.005, 0.001),
    num_epochs=10,
    runs=1,
    verbose=True,
):
    """
    grid search over learning rate values

    Args:
        data: input data
        model: model to train
        loss_fun: training loss takes model and data as input
        validation_loss_fun: function to compute validation loss input: (model, data)
        lr_grid: learning rate values to try
        num_epochs: number of epochs for training
        runs: number of training runs to average over for selecting best learning rate
        verbose: if ``True``, output training progress

    Returns:
        best learning rate, validation loss for all runs
    """
    val_loss = torch.zeros((len(lr_grid), runs))
    for i, lr in enumerate(lr_grid):
        for r in range(runs):
            model.reset_parameters()
            model = train(
                data, model, loss_fun, num_epochs=num_epochs, lr=lr, verbose=verbose
            )
            val_loss[i, r] = validation_loss_fun(model, data)
    model.reset_parameters()
    return lr_grid[torch.argmax(torch.mean(val_loss, 1))], val_loss


def train(
    data,
    model,
    loss_fun,
    num_epochs=10000,
    patience=20,
    lr=0.01,
    weight_decay=0.0,
    verbose=True,
    logger=lambda loss: None,
):
    """
    train an embedding model

    Args:
        data: network data


        model: embedding auto-encoder model

        loss_fun: loss function to use with model (takes arguments ``model``, ``data``)

        num_epochs: number of training epochs

        patience: patience for early stopping

        lr: learining rate (default: 0.01)

        weight_decay: weight decay for optimizer (default: 0.0)

        verbose: if ``True``, display training progress (default: ``True``)

        logger: function that receives the training loss as input and
            is called after each epoch (does nothing by default)

    Returns:
        trained model

    This function uses the Adam optimizer for training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    with EarlyStopping(patience) as stop:
        for e in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fun(model, data)
            f_loss = float(loss)
            logger(f_loss)
            if verbose:
                print(f"epoch {e}: loss={f_loss}")
            if stop(f_loss, model):
                if verbose:
                    print(f"Early stopping at epoch {e}")
                break
            loss.backward()
            optimizer.step()
    return model
