# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import numpy as np


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
