# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
)
from WGFormer.data import (
    KeyDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    ConformerSampleConfGDataset,
    data_utils,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


@register_task("WGFormer")
class WGFormerTask(UnicoreTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data", 
            help="downstream data path"
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        
        idx_dataset = KeyDataset(dataset, "idx")
        src_dataset = KeyDataset(dataset, "atoms")
        
        sample_dataset = ConformerSampleConfGDataset(
                dataset, self.args.seed, "atoms", "coordinates", "target"
            )
        
        sample_dataset = NormalizeDataset(sample_dataset, "coordinates")
        sample_dataset = NormalizeDataset(sample_dataset, "target")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        
        coord_dataset = KeyDataset(sample_dataset, "coordinates")
        tgt_coord_dataset = KeyDataset(sample_dataset, "target")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        tgt_coord_dataset = FromNumpyDataset(tgt_coord_dataset)
        tgt_coord_dataset = PrependAndAppend(tgt_coord_dataset, 0.0, 0.0)
        tgt_distance_dataset = DistanceDataset(tgt_coord_dataset)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
                
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "coord_target": RightPadDatasetCoord(
                        tgt_coord_dataset,
                        pad_idx=0,
                    ),
                    "distance_target": RightPadDataset2D(
                        tgt_distance_dataset,
                        pad_idx=0,
                    ),
                },
                "idx_name": RawArrayDataset(idx_dataset),
            },
        )
        if split.startswith("train"):
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model
