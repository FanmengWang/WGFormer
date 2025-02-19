# WGFormer

Installation
------------
* Our code has been tested in the following environment:
  | Package | Version |
  | --- | --- |
  | Python | 3.8.13 |
  | PyTorch | 1.11.0 |
  | CUDA | 11.3.1 |
  | RDKit | 2022.9.5 |
* Run the following to install with the necessary dependencies.
  ```bash
  conda env create -f env.yml
  conda activate WGFormer
  ```

Data Preparation
------------
We use the Molecule3D and QM9 provided in [GTMGC](https://github.com/Rich-XGK/GTMGC) as our datasets.



Training
------------
*
  ```bash
  bash train.sh
  ```


Inference
------------
*
  ```bash
  bash inference.sh valid
  bash inference.sh test
  ```


Evaluation
------------
*
  ```bash
  python eval.py valid
  python eval.py test
  ```
