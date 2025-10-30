# holocron.optim

To use `holocron.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

## Optimizers

Implementations of recent parameter optimizer for Pytorch modules.

::: holocron.optim.LARS
    options:
        heading_level: 3
        members: no

::: holocron.optim.LAMB
    options:
        heading_level: 3
        members: no

::: holocron.optim.RaLars
    options:
        heading_level: 3
        members: no

::: holocron.optim.TAdam
    options:
        heading_level: 3
        members: no

::: holocron.optim.AdaBelief
    options:
        heading_level: 3
        members: no

::: holocron.optim.AdamP
    options:
        heading_level: 3
        members: no

::: holocron.optim.Adan
    options:
        heading_level: 3
        members: no

::: holocron.optim.AdEMAMix
    options:
        heading_level: 3
        members: no


## Optimizer wrappers

`holocron.optim` also implements optimizer wrappers.

A base optimizer should always be passed to the wrapper; e.g., you
should write your code this way:

```python
optimizer = ...
optimizer = wrapper(optimizer)
```

::: holocron.optim.Lookahead
    options:
        heading_level: 3
        members: no

::: holocron.optim.Scout
    options:
        heading_level: 3
        members: no
