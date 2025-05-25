# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import pickle
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from copy import deepcopy
from rdkit import RDLogger
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

_RANDOM_URLS = {
    "train": [
        f"./Molecule3D/Random/random_train_0.sdf",
        f"./Molecule3D/Random/random_train_1.sdf",
        f"./Molecule3D/Random/random_train_2.sdf",
        f"./Molecule3D/Random/random_train_3.sdf",
        f"./Molecule3D/Random/random_train.csv",
    ],
    "valid": [
        f"./Molecule3D/Random/random_valid.sdf",
        f"./Molecule3D/Random/random_valid.csv",
    ],
    "test": [
        f"./Molecule3D/Random/random_test.sdf",
        f"./Molecule3D/Random/random_test.csv",
    ],
}

_SCAFFOLD_URLS = {
    "train": [
        f"./Molecule3D/Scaffold/scaffold_train_0.sdf",
        f"./Molecule3D/Scaffold/scaffold_train_1.sdf",
        f"./Molecule3D/Scaffold/scaffold_train_2.sdf",
        f"./Molecule3D/Scaffold/scaffold_train_3.sdf",
        f"./Molecule3D/Scaffold/scaffold_train.csv",
    ],
    "valid": [
        f"./Molecule3D/Scaffold/scaffold_valid.sdf",
        f"./Molecule3D/Scaffold/scaffold_valid.csv",
    ],
    "test": [
        f"./Molecule3D/Scaffold/scaffold_test.sdf",
        f"./Molecule3D/Scaffold/scaffold_test.csv",
    ],
}


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

    splitting_strategy_list = ['Random', 'Scaffold']
    
    for splitting_strategy in splitting_strategy_list:

        if splitting_strategy == 'Random':
            URLS = _RANDOM_URLS
        elif splitting_strategy == 'Scaffold':
            URLS = _SCAFFOLD_URLS
        
        split_list = ['train', 'valid', 'test']
        
        for split_name in split_list:
            env = lmdb.open(
                f'./Molecule3D/{splitting_strategy}/{split_name}.lmdb',
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(100e9),
            )
            txn = env.begin(write=True)

            archives = URLS[split_name]
            
            if split_name == 'train':
                sdf_files, csv_file = archives[:-1], archives[-1]
                suppliers = [Chem.SDMolSupplier(fileName=file, removeHs=False, sanitize=True) for file in sdf_files]
                properties_df = pd.read_csv(csv_file)
                valid_num = 0
                for idx, row_series in properties_df.iterrows():
                    row, col = idx // 600000, idx % 600000
                    mol = suppliers[row][col]
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
            else:
                sdf_file, csv_file = archives[0], archives[1]
                suppliers = Chem.SDMolSupplier(fileName=sdf_file, removeHs=False, sanitize=True)
                properties_df = pd.read_csv(csv_file)
                valid_num = 0
                for idx, row_series in properties_df.iterrows():
                    mol = suppliers[idx]
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
            print(f'Molecule3D-{splitting_strategy}-{split_name}: valid_num:{valid_num}')