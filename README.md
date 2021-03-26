## Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image

This repository contains the code that accompanies our CVPR 2020 paper
[Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image](https://superquadrics.com/hierarchical-primitives.html)

![Teaser](img/teaser.png)

You can find detailed usage instructions for training your own models and using our pretrained models below.

If you found this work influential or helpful for your research, please consider citing

```
@Inproceedings{Paschalidou2020CVPR,
     title = {Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image},
     author = {Paschalidou, Despoina and Luc van Gool and Geiger, Andreas},
     booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
     year = {2020}
}
```

## Installation & Dependencies

Our codebase has the following dependencies:

- [numpy](https://numpy.org/doc/stable/user/install.html)
- [cython](https://cython.readthedocs.io/en/latest/src/quickstart/build.html)
- [pyquaternion](http://kieranwynn.github.io/pyquaternion/)
- [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [pykdtree](https://github.com/storpipfugl/pykdtree)
- [torch && torchvision](https://pytorch.org/get-started/locally/)
- [trimesh](https://github.com/mikedh/trimesh)

For the visualizations, we use [simple-3dviz](http://simple-3dviz.com), which
is our easy-to-use library for visualizing 3D data using Python and ModernGL and
[matplotlib](https://matplotlib.org/) for the colormaps. Note that
[simple-3dviz](http://simple-3dviz.com) provides a lightweight and easy-to-use
scene viewer using [wxpython](https://www.wxpython.org/). If you wish you use
our scripts for visualizing the reconstructed primitives, you will need to also
install [wxpython](https://anaconda.org/anaconda/wxpython).

The simplest way to make sure that you have all dependencies in place is to use
[conda](https://docs.conda.io/projects/conda/en/4.6.1/index.html). You can
create a conda environment called ```hierarchical_primitives``` using
```
conda env create -f environment.yaml
conda activate hierarchical_primitives
```

Next compile the extenstion modules. You can do this via
```
python setup.py build_ext --inplace
pip install -e .
```

## Usage

As soon as you have installed all dependencies you can now start training new
models from scratch, evaluate our pre-trained models and visualize the
recovered primitives using one of our pre-trained models.

### Reconstruction
To visualize the predicted primitives using a trained model, we provide the
``visualize_predictions.py`` script. In particular, it performs the forward
pass and visualizes the predicted primitives using
[simple-3dviz](https://simple-3dviz.com/). To execute it simply run
To run the ``visualize_predictions.py`` script you need to run
```
python visualize_predictions.py path_to_config_yaml path_to_output_dir --weight_file path_to_weight_file --model_tag MODEL_TAG --from_fit
```
where the argument ``--weight_file`` specifies the path to a trained model and
the argument ``--model_tag`` defines the model_tag of the input to be
reconstructed.

### Hierarchy Reconstruction

### Training
Finally, to train a new network from scratch, we provide the
``train_network.py`` script. To execute this script, you need to specify the
path to the configuration file you wish to use and the path to the output
directory, where the trained models and the training statistics will be saved.
Namely, to train a new model from scratch, you simply need to run
```
python train_network.py path_to_config_yaml path_to_output_dir
```
Note tha it is also possible to start from a previously trained model by
specifying the ``--weight_file`` argument, which should contain the path to a
previously trained model. Furthermore, by using the arguments `--model_tag` and
``--category_tag``, you can also train your network on a particular model (e.g.
a specific plane, car, human etc.) or a specific object category (e.g. planes,
chairs etc.).

Also make sure to update the ``dataset_directory`` argument in the provided
config file based on the path where your dataset is stored.

## Contribution

Contributions such as bug fixes, bug reports, suggestions etc. are more than
welcome and should be submitted in the form of new issues and/or pull requests
on Github.

## License

Our code is released under the MIT license which practically allows anyone to do anything with it.
MIT license found in the LICENSE file.

## Relevant Research

Below we list some papers that are relevant to our work.

**Ours:**
- Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks [pdf](https://arxiv.org/pdf/2103.10429.pdf)
- Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image [pdf](https://paschalidoud.github.io/)
- Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids [pdf](https://arxiv.org/pdf/1904.09970.pdf) [blog](https://autonomousvision.github.io/superquadrics-revisited/)

**By Others:**
- Learning Shape Abstractions by Assembling Volumetric Primitives [pdf](https://arxiv.org/pdf/1612.00404.pdf)
- 3D-PRNN: Generating Shape Primitives with Recurrent Neural Networks [pdf](https://arxiv.org/abs/1708.01648.pdf)
- Im2Struct: Recovering 3D Shape Structure From a Single RGB Image [pdf](http://openaccess.thecvf.com/content_cvpr_2018/html/Niu_Im2Struct_Recovering_3D_CVPR_2018_paper.pdf)
- Learning shape templates with structured implicit functions [pdf](https://arxiv.org/abs/1904.06447)
- CvxNet: Learnable Convex Decomposition [pdf](https://arxiv.org/abs/1909.05736)

Below we also list some more papers that are more closely related to superquadrics
- Equal-Distance Sampling of Supercllipse Models [pdf](https://pdfs.semanticscholar.org/3e6f/f812b392f9eb70915b3c16e7bfbd57df379d.pdf)
- Revisiting Superquadric Fitting: A Numerically Stable Formulation [link](https://ieeexplore.ieee.org/document/8128485)
- Segmentation and Recovery of Superquadric Models using Convolutional Neural Networks [pdf](https://arxiv.org/abs/2001.10504)
