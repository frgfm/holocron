# ReXNet

The ResNet model is based on the ["ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"](https://arxiv.org/pdf/2007.00992.pdf) paper.

## Architecture overview

This paper investigates the effect of channel configuration in convolutional bottlenecks.

The key takeaways from the paper are the following:

- increasing the depth ratio of conv 1x1 and inverted bottlenecks
- replace ReLU6 with SiLU


## Model builders

The following model builders can be used to instantiate a ReXNet model, with or
without pre-trained weights. All the model builders internally rely on the
[`ReXNet`][holocron.models.ReXNet] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ReXNet
            - rexnet1_0x
            - rexnet1_3x
            - rexnet1_5x
            - rexnet2_0x
            - rexnet2_2x
