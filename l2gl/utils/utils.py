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
"""TODO: module docstring for utils.py"""

from tempfile import TemporaryFile
from time import perf_counter
import torch
import torch.nn


def speye(n: int, dtype: torch.dtype = torch.float) -> torch.Tensor:
    """identity matrix of dimension n as sparse_coo_tensor."""
    return torch.sparse_coo_tensor(
        torch.tile(torch.arange(n, dtype=torch.long), (2, 1)),
        torch.ones(n, dtype=dtype),
        (n, n),
    )


def get_device(model: torch.nn.Module) -> torch.device:
    """TODO: docstring for get_device."""
    return next(model.parameters()).device


def set_device(device: str | None = None):
    """TODO: docstring for set_device."""
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(device)



class Timer:
    """
    Context manager for accumulating execution time

    Adds the time taken within block to a running total.

    """

    def __init__(self):
        self.total: float = 0.0
        self.tic: float | None = None

    def __enter__(self):
        self.tic = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.tic is not None:
            self.total += perf_counter() - self.tic


def flatten(lst, ltypes=(list, tuple)):
    """TODO: docstring for flatten."""
    if isinstance(lst, ltypes):
        ltype = type(lst)
        lst = list(lst)
        i = 0
        while i < len(lst):
            while isinstance(lst[i], ltypes):
                if not lst[i]:
                    lst.pop(i)
                    i -= 1
                    break

                lst[i : i + 1] = lst[i]
            i += 1
        return ltype(lst)

    return lst
