"""Running mean of arrays."""

import numpy as np


class RunningStatsArray:
    """Running (incremental) mean of equally shaped arrays, so that grand averages can be
    computed without keeping all arrays in memory. The current mean is `self.mean`, the
    number of arrays pushed so far `self.n`."""

    def __init__(self):
        """Start with no arrays (`mean` is None until the first `push`)."""
        self.n = 0
        self.mean = None

    def push(self, x):
        """Update the running mean with a new array x."""

        x = np.asarray(x)
        if self.mean is None:
            self.mean = np.zeros_like(x)

        self.n += 1
        old_weight = (self.n - 1) / self.n
        new_weight = 1 / self.n

        self.mean = old_weight * self.mean + new_weight * x
