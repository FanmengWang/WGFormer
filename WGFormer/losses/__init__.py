# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from pathlib import Path

for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("WGFormer.losses." + file.name[:-3])
