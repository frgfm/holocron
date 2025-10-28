# RepVGG

The ResNet model is based on the ["RepVGG: Making VGG-style ConvNets Great Again"](https://arxiv.org/pdf/2101.03697.pdf) paper.

## Architecture overview

This paper revisits the VGG architecture by adapting its parameter setting in training and inference mode to combine the original VGG speed and the block design of ResNet.

![RepVGG architecture](https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg.png)

The key takeaways from the paper are the following:

- have different block architectures between training and inference modes
- the block is designed in a similar fashion as a ResNet bottleneck but in a way that all branches can be fused into a single one
- The more complex training architecture improves gradient flow and overall optimization, while its inference counterpart is optimized for minimum latency and memory usage


## Model builders

The following model builders can be used to instantiate a RepVGG model, with or
without pre-trained weights. All the model builders internally rely on the
`holocron.models.classification.revpgg.RepVGG` base class. Please refer to the [source
code](https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/repvgg.py) for
more details about this class.

::: holocron.models.classification.repvgg
    options:
        heading_level: 3
        show_root_heading: false
        members:
            - repvgg_a0
            - repvgg_a1
            - repvgg_a2
            - repvgg_b0
            - repvgg_b1
            - repvgg_b2
            - revpgg_b3
