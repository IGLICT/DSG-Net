
Please fill in [this form](https://forms.gle/vs1B6myBHUfebqGG8) to download the data.

This zip file provides the processed PartNet hierarchy of graphs data for six object categories used in the paper: chair, storage furniture, table, lamp, synthetic data.

## About this repository

```
    chair_hier/                         # hierarchy of graphs data for chairs
            [PartNet_anno_id].json      # storing the tree structure, detected sibling edges,
                                        # and all part oriented bounding box parameters) for a chair
    chair_dgeo/                         # part geometry for chairs
            [PartNet_anno_id].npz       # storing all part geometry (deformation gradients) for a chair
    cube_meshinfo.mat                   # storing some geometry information of a cube, recovering the deformation
                                        # gradient of a shape to coordinate representation by the file

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
