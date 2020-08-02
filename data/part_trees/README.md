

This zip file provides the processed PartNet hierarchy of graphs data for five object categories used in the paper: chair, storage furniture, table, lamp, synthetic data.

## About this repository

```
    Chair_all_no_other_less_than_10_parts-train/                                        # hierarchy of graphs data for chairs and training anno id
                                                pt-0/
                                                    *.txt                               # useless
                                                    template.json                       # storing the tree structure, detected sibling edges,
                                                                                        # without any parameters, only tree structure
                                                info.txt                                # training list, the folders pt-xx, ..., pt-xx coorespond
                                                                                        # to the order of content of info.txt
    Chair_all_no_other_less_than_10_parts-test/                                         # hierarchy of graphs data for chairs and test anno id
                                                pt-0/
                                                    *.txt                               # useless
                                                    template.json                       # storing the tree structure, detected sibling edges,
                                                                                        # without any parameters, only tree structure
                                                info.txt                                # test list, the folders pt-xx, ..., pt-xx coorespond
                                                                                        # to the order of content of info.txt. Subsets of data used
                                                                                        # in StructureNet where all parts are labeled and no more than
                                                                                        # 10 parts per parent node. We use this subset for StructureNet

```

## Cite

Please cite both [PartNet](https://cs.stanford.edu/~kaichun/partnet/) and StructureNet if you use this data.


    @InProceedings{Mo_2019_CVPR,
        author = {Mo, Kaichun and Zhu, Shilin and Chang, Angel X. and Yi, Li and Tripathi, Subarna and Guibas, Leonidas J. and Su, Hao},
        title = {{PartNet}: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level {3D} Object Understanding},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }
