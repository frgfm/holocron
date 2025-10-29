# holocron.optim

To use `holocron.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

## Optimizers

Implementations of recent parameter optimizer for Pytorch modules.

::: holocron.optim
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - LARS
            - LAMB
            - RaLars
            - TAdam
            - AdaBelief
            - AdamP
            - Adan
            - AdEMAMix

## Optimizer wrappers

`holocron.optim` also implements optimizer wrappers.

A base optimizer should always be passed to the wrapper; e.g., you
should write your code this way:

```python
>>> optimizer = ...
>>> optimizer = wrapper(optimizer)
```

::: holocron.optim
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - Lookahead
            - Scout
