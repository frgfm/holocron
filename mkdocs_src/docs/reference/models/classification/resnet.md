# ResNet

The ResNet model is based on the ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) paper.

## Architecture overview

This paper introduces a few tricks to maximize the depth of convolutional architectures that can be trained.

The key takeaways from the paper are the following:

* add a shortcut connection in bottleneck blocks to ease the gradient flow
* extensive use of batch normalization layers


## Model builders

The following model builders can be used to instantiate a ResNeXt model, with or
without pre-trained weights. All the model builders internally rely on the
[`ResNet`][holocron.models.ResNet] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ResNet
            - resnet18
            - resnet34
            - resnet50
            - resnet50d
            - resnet101
            - resnet152
