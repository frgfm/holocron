# DarkNetV4

The DarkNetV4 model is based on the ["CSPNet: A New Backbone that can Enhance Learning Capability of CNN"](https://arxiv.org/pdf/1911.11929.pdf) paper.

## Architecture overview

This paper makes a more powerful version than its predecedors by increasing depth and using ResNet tricks.

The key takeaways from the paper are the following:

- add cross-path connections to its predecessors
- explores newer non-linearities


## Model builders

The following model builders can be used to instantiate a DarknetV3 model, with or
without pre-trained weights. All the model builders internally rely on the
[`DarknetV4`][holocron.models.DarknetV4] base class.

::: holocron.models.classification
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DarknetV4
            - cspdarknet53
            - cspdarknet53_mish
