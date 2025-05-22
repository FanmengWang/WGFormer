# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import sys
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
from rdkit import Chem
from copy import deepcopy
from rdkit import RDLogger
from rdkit.Chem import rdchem, rdMolAlign

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def get_metrics(mol: rdchem.Mol, mol_h: rdchem.Mol = None, R: np.ndarray = None, R_h: np.ndarray = None, removeHs: bool = False):
    mol_h = deepcopy(mol)
    R_h = R_h.tolist()
    conf_h = rdchem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf_h.SetAtomPosition(i, R_h[i])
    mol_h.RemoveConformer(0)
    mol_h.AddConformer(conf_h)
                
    R, R_h = mol.GetConformer().GetPositions(), mol_h.GetConformer().GetPositions()
    R, R_h = torch.from_numpy(R), torch.from_numpy(R_h)
    D, D_h = torch.cdist(R, R), torch.cdist(R_h, R_h)
    mae = F.l1_loss(D, D_h, reduction="sum").item()
    mse = F.mse_loss(D, D_h, reduction="sum").item()
    num_dist = D.numel()
    
    if removeHs:
        try:
            mol, mol_h = Chem.RemoveHs(mol), Chem.RemoveHs(mol_h)
        except Exception as e:
            pass
    
    rmsd = rdMolAlign.GetBestRMS(mol, mol_h)
    return {
        "mae": mae,
        "mse": mse,
        "rmsd": rmsd,
        "num_dist": num_dist,
    }
    
    
if __name__ == '__main__':
    
    split_name = sys.argv[1]

    supplier = Chem.SDMolSupplier(fileName='./data/QM9/gdb9.sdf', removeHs=False, sanitize=True)

    input_file = f'./infer_results_QM9/checkpoints_QM9_{split_name}.out.pkl'
    
    predicts = pd.read_pickle(input_file)
    
    total_mae, total_mse, total_dist, total_rmsd = 0.0, 0.0, 0.0, 0.0
    
    num_mol = 0
    
    for epoch in range(len(predicts)):
        predict = predicts[epoch]
        bsz = predicts[epoch]['bsz']
        for index in range(bsz):
            idx = int(predict['idx_name'][index])
            coord_target = predict['coord_target'][index].detach().cpu().numpy()
            coord_predict = predict['coord_predict'][index].detach().cpu().numpy()
            coord_mask = np.any(coord_target != coord_target[0], axis=1)
            
            mol = supplier[idx]    
            metrics = get_metrics(mol=mol, R_h=coord_predict[coord_mask], removeHs=True)
            mae, mse, rmsd, num_dist = metrics["mae"], metrics["mse"], metrics["rmsd"], metrics["num_dist"]
            total_mae += mae
            total_mse += mse
            total_dist += num_dist
            total_rmsd += rmsd
            num_mol += 1
    
    mae = total_mae / total_dist
    rmse = np.sqrt(total_mse / total_dist)
    rmsd = total_rmsd / num_mol
    
    print(input_file)
    print(f"D-MAE: {mae}, D-RMSE: {rmse}, C-RMSD: {rmsd}")
    print(f'***************************************')
    
  