# PyConvResNet

The PyConvResNet model is based on the ["Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"](https://arxiv.org/pdf/2006.11538.pdf) paper.

## Architecture overview

This paper explores an alternative approach for convolutional block in a pyramidal fashion.

![PyConvResNet architecture](https://github.com/frgfm/Holocron/releases/download/v0.2.1/pyconv_resnet.png)

The key takeaways from the paper are the following:

- replaces standard convolutions with pyramidal convolutions
- extends kernel size while increasing group size to balance the number of operations


## Model builders

The following model builders can be used to instantiate a PyConvResNet model, with or
without pre-trained weights. All the model builders internally rely on the
[`ResNet`][holocron.models.ResNet] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - pyconv_resnet50
            - pyconvhg_resnet50
