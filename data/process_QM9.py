# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from rdkit import RDLogger
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def extract_coords(mol, num_atoms, split_name):
    coordinate_list=[]
    if split_name == 'train':
        for seed in range(11):
            mol_seed = deepcopy(mol)   
            mol_seed.RemoveAllConformers()     
            res = AllChem.EmbedMolecule(mol_seed, randomSeed=seed)  
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol_seed)
                    coordinates = mol_seed.GetConformer().GetPositions()
                except:
                    AllChem.Compute2DCoords(mol_seed)
                    coordinates = mol_seed.GetConformer().GetPositions()
            else:
                AllChem.Compute2DCoords(mol_seed)
                coordinates = mol_seed.GetConformer().GetPositions()
            coordinates = coordinates[:num_atoms]
            coordinate_list.append(coordinates.astype(np.float32))
    else:
        mol = deepcopy(mol)  
        mol.RemoveAllConformers()      
        res = AllChem.EmbedMolecule(mol, useRandomCoords=False)  
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions()
            except:
                AllChem.Compute2DCoords(mol)
                coordinates = mol.GetConformer().GetPositions()           
        else:
            AllChem.Compute2DCoords(mol)
            coordinates = mol.GetConformer().GetPositions()
        coordinates = coordinates[:num_atoms]
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


if __name__ == '__main__':
    
    supplier = Chem.SDMolSupplier(fileName='./QM9/gdb9.sdf', removeHs=False, sanitize=True)
    
    split_list = ['train', 'valid', 'test']
    
    for split_name in split_list:
        env = lmdb.open(
            f'./QM9/{split_name}.lmdb',
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn = env.begin(write=True)
    
        valid_num = 0
        with open(f'./QM9/{split_name}_indices.csv', 'r') as f:
            next(f)
            for line in tqdm(f.readlines()):
                idx = int(line.strip())
                mol = supplier[idx]
                if mol is None:
                    continue
                data_info = {}
                data_info['idx'] = idx
                data_info['target'] = mol.GetConformer().GetPositions()
                data_info['atoms'] = [atom.GetSymbol() for atom in mol.GetAtoms()]
                mol = AllChem.AddHs(mol)
                data_info['coordinates'] = extract_coords(mol, len(data_info['target']), split_name)
                txn.put(f'{valid_num}'.encode("ascii"), pickle.dumps(data_info, protocol=-1))
                valid_num += 1
    
        txn.commit()
        env.close()
        print(f'QM9-{split_name}: valid_num:{valid_num}')

