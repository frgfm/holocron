# TridentNet

The ResNeXt model is based on the ["Scale-Aware Trident Networks for Object Detection"](https://arxiv.org/pdf/1901.01892.pdf) paper.

## Architecture overview

This paper replaces the bottleneck block of ResNet architectures by a multi-scale version.

![TridentNet architecture](https://github.com/frgfm/Holocron/releases/download/v0.2.1/tridentnet.png)

The key takeaways from the paper are the following:

- switch bottleneck to a 3 branch system
- all parallel branches share the same parameters but using different dilation values


## Model builders

The following model builders can be used to instantiate a TridentNet model, with or
without pre-trained weights. All the model builders internally rely on the
[`ResNet`][holocron.models.ResNet] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - tridentnet50
