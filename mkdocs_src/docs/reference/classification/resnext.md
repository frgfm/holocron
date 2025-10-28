# ResNeXt

The ResNeXt model is based on the ["Aggregated Residual Transformations for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431.pdf) paper.

## Architecture overview

This paper improves the ResNet architecture by increasing the width of bottleneck blocks

The key takeaways from the paper are the following:

- increases the number of channels in bottlenecks
- switches to group convolutions to balance the number of operations


## Model builders

The following model builders can be used to instantiate a ResNet model, with or
without pre-trained weights. All the model builders internally rely on the
`holocron.models.classification.resnet.ResNet` base class. Please refer to the [source
code](https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/resnet.py) for
more details about this class.

::: holocron.models.classification.resnet
    options:
        heading_level: 3
        show_root_heading: false
        members:
            - resnext50_32x4d
            - resnext101_32x8d
