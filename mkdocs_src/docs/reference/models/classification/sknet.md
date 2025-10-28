# SKNet

The ResNet model is based on the ["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf) paper.

## Architecture overview

This paper revisits the concept of having a dynamic receptive field selection in convolutional blocks.

![SKNet architecture](https://github.com/frgfm/Holocron/releases/download/v0.2.1/skconv.png)

The key takeaways from the paper are the following:

- performs convolutions with multiple kernel sizes
- implements a cross-channel attention mechanism


## Model builders

The following model builders can be used to instantiate a SKNet model, with or
without pre-trained weights. All the model builders internally rely on the
`holocron.models.classification.resnet.ResNet` base class. Please refer to the [source
code](https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/sknet.py) for
more details about this class.

::: holocron.models.classification.sknet
    options:
        heading_level: 3
        show_root_heading: false
        members:
            - sknet50
            - sknet101
            - sknet152
