# holocron.nn

An addition to the `torch.nn` module of Pytorch to extend the range of neural networks building blocks.

## Non-linear activations

### HardMish

```python
HardMish(inplace: bool = False)
```

Bases: `_Activation`

Implements the Hard Mish activation module from ["H-Mish"](https://github.com/digantamisra98/H-Mish).

This activation is computed as follows:

[ f(x) = \\frac{x}{2} \\cdot \\min(2, \\max(0, x + 2)) ]

| PARAMETER | DESCRIPTION                                                                     |
| --------- | ------------------------------------------------------------------------------- |
| `inplace` | should the operation be performed inplace **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/nn/modules/activation.py`

```python
def __init__(self, inplace: bool = False) -> None:
    super().__init__()
    self.inplace: bool = inplace
```

### NLReLU

```python
NLReLU(beta: float = 1.0, inplace: bool = False)
```

Bases: `_Activation`

Implements the Natural-Logarithm ReLU activation module from ["Natural-Logarithm-Rectified Activation Function in Convolutional Neural Networks"](https://arxiv.org/pdf/1908.03682.pdf).

This activation is computed as follows:

[ f(x) = ln(1 + \\beta \\cdot max(0, x)) ]

| PARAMETER | DESCRIPTION                                                                     |
| --------- | ------------------------------------------------------------------------------- |
| `beta`    | beta used for NReLU **TYPE:** `float` **DEFAULT:** `1.0`                        |
| `inplace` | should the operation be performed inplace **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/nn/modules/activation.py`

```python
def __init__(self, beta: float = 1.0, inplace: bool = False) -> None:
    super().__init__(inplace)
    self.beta: float = beta
```

### FReLU

```python
FReLU(in_channels: int, kernel_size: int = 3)
```

Bases: `Module`

Implements the Funnel activation module from ["Funnel Activation for Visual Recognition"](https://arxiv.org/pdf/2007.11824.pdf).

This activation is computed as follows:

[ f(x) = max(\\mathbb{T}(x), x) ]

where the (\\mathbb{T}) is the spatial contextual feature extraction. It is a convolution filter of size `kernel_size`, same padding and groups equal to the number of input channels, followed by a batch normalization.

| PARAMETER     | DESCRIPTION                                                     |
| ------------- | --------------------------------------------------------------- |
| `in_channels` | number of input channels **TYPE:** `int`                        |
| `kernel_size` | size of the convolution filter **TYPE:** `int` **DEFAULT:** `3` |

Source code in `holocron/nn/modules/activation.py`

```python
def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
    self.bn = nn.BatchNorm2d(in_channels)
```

## Loss functions

### Loss

```python
Loss(weight: float | list[float] | Tensor | None = None, ignore_index: int = -100, reduction: str = 'mean')
```

Bases: `Module`

Base loss class.

| PARAMETER      | DESCRIPTION                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `weight`       | class weight for loss computation **TYPE:** \`float                                                          |
| `ignore_index` | specifies target value that is ignored and do not contribute to gradient **TYPE:** `int` **DEFAULT:** `-100` |
| `reduction`    | type of reduction to apply to the final loss **TYPE:** `str` **DEFAULT:** `'mean'`                           |

| RAISES                | DESCRIPTION                              |
| --------------------- | ---------------------------------------- |
| `NotImplementedError` | if the reduction method is not supported |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(
    self,
    weight: float | list[float] | Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> None:
    super().__init__()
    # Cast class weights if possible
    self.weight: Tensor | None
    if isinstance(weight, (float, int)):
        self.register_buffer("weight", torch.Tensor([weight, 1 - weight]))
    elif isinstance(weight, list):
        self.register_buffer("weight", torch.Tensor(weight))
    elif isinstance(weight, Tensor):
        self.register_buffer("weight", weight)
    else:
        self.weight: Tensor | None = None
    self.ignore_index: int = ignore_index
    # Set the reduction method
    if reduction not in {"none", "mean", "sum"}:
        raise NotImplementedError("argument reduction received an incorrect input")
    self.reduction: str = reduction
```

### FocalLoss

```python
FocalLoss(gamma: float = 2.0, **kwargs: Any)
```

Bases: `Loss`

Implementation of Focal Loss as described in ["Focal Loss for Dense Object Detection"](https://arxiv.org/pdf/1708.02002.pdf).

While the weighted cross-entropy is described by:

[ CE(p_t) = -\\alpha_t log(p_t) ]

where (\\alpha_t) is the loss weight of class (t), and (p_t) is the predicted probability of class (t).

the focal loss introduces a modulating factor

[ FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma log(p_t) ]

where (\\gamma) is a positive focusing parameter.

| PARAMETER  | DESCRIPTION                                                               |
| ---------- | ------------------------------------------------------------------------- |
| `gamma`    | exponent parameter of the focal loss **TYPE:** `float` **DEFAULT:** `2.0` |
| `**kwargs` | keyword args of Loss **TYPE:** `Any` **DEFAULT:** `{}`                    |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(self, gamma: float = 2.0, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.gamma: float = gamma
```

### MultiLabelCrossEntropy

```python
MultiLabelCrossEntropy(*args: Any, **kwargs: Any)
```

Bases: `Loss`

Implementation of the cross-entropy loss for multi-label targets

| PARAMETER  | DESCRIPTION                                            |
| ---------- | ------------------------------------------------------ |
| `*args`    | args of Loss **TYPE:** `Any` **DEFAULT:** `()`         |
| `**kwargs` | keyword args of Loss **TYPE:** `Any` **DEFAULT:** `{}` |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
```

### ComplementCrossEntropy

```python
ComplementCrossEntropy(gamma: float = -1, **kwargs: Any)
```

Bases: `Loss`

Implements the complement cross entropy loss from ["Imbalanced Image Classification with Complement Cross Entropy"](https://arxiv.org/pdf/2009.02189.pdf)

| PARAMETER  | DESCRIPTION                                            |
| ---------- | ------------------------------------------------------ |
| `gamma`    | smoothing factor **TYPE:** `float` **DEFAULT:** `-1`   |
| `**kwargs` | keyword args of Loss **TYPE:** `Any` **DEFAULT:** `{}` |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(self, gamma: float = -1, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.gamma: float = gamma
```

### MutualChannelLoss

```python
MutualChannelLoss(weight: float | list[float] | Tensor | None = None, ignore_index: int = -100, reduction: str = 'mean', xi: int = 2, alpha: float = 1)
```

Bases: `Loss`

Implements the mutual channel loss from ["The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"](https://arxiv.org/pdf/2002.04264.pdf).

| PARAMETER      | DESCRIPTION                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `weight`       | class weight for loss computation **TYPE:** \`float                                                          |
| `ignore_index` | specifies target value that is ignored and do not contribute to gradient **TYPE:** `int` **DEFAULT:** `-100` |
| `reduction`    | type of reduction to apply to the final loss **TYPE:** `str` **DEFAULT:** `'mean'`                           |
| `xi`           | num of features per class **TYPE:** `int` **DEFAULT:** `2`                                                   |
| `alpha`        | diversity factor **TYPE:** `float` **DEFAULT:** `1`                                                          |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(
    self,
    weight: float | list[float] | Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    xi: int = 2,
    alpha: float = 1,
) -> None:
    super().__init__(weight, ignore_index, reduction)
    self.xi: int = xi
    self.alpha: float = alpha
```

### DiceLoss

```python
DiceLoss(weight: float | list[float] | Tensor | None = None, gamma: float = 1.0, eps: float = 1e-08)
```

Bases: `Loss`

Implements the dice loss from ["V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"](https://arxiv.org/pdf/1606.04797.pdf).

| PARAMETER | DESCRIPTION                                                                        |
| --------- | ---------------------------------------------------------------------------------- |
| `weight`  | class weight for loss computation **TYPE:** \`float                                |
| `gamma`   | recall/precision control param **TYPE:** `float` **DEFAULT:** `1.0`                |
| `eps`     | small value added to avoid division by zero **TYPE:** `float` **DEFAULT:** `1e-08` |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(
    self,
    weight: float | list[float] | Tensor | None = None,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> None:
    super().__init__(weight)
    self.gamma: float = gamma
    self.eps: float = eps
```

### PolyLoss

```python
PolyLoss(*args: Any, eps: float = 2.0, **kwargs: Any)
```

Bases: `Loss`

Implements the Poly1 loss from ["PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"](https://arxiv.org/pdf/2204.12511.pdf).

| PARAMETER  | DESCRIPTION                                                   |
| ---------- | ------------------------------------------------------------- |
| `*args`    | args of Loss **TYPE:** `Any` **DEFAULT:** `()`                |
| `eps`      | epsilon 1 from the paper **TYPE:** `float` **DEFAULT:** `2.0` |
| `**kwargs` | keyword args of Loss **TYPE:** `Any` **DEFAULT:** `{}`        |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(
    self,
    *args: Any,
    eps: float = 2.0,
    **kwargs: Any,
) -> None:
    super().__init__(*args, **kwargs)
    self.eps: float = eps
```

## Loss wrappers

### ClassBalancedWrapper

```python
ClassBalancedWrapper(criterion: Module, num_samples: Tensor, beta: float = 0.99)
```

Bases: `Module`

Implementation of the class-balanced loss as described in ["Class-Balanced Loss Based on Effective Number of Samples"](https://arxiv.org/pdf/1901.05555.pdf).

Given a loss function (\\mathcal{L}), the class-balanced loss is described by:

[ CB(p, y) = \\frac{1 - \\beta}{1 - \\beta^{n_y}} \\mathcal{L}(p, y) ]

where (p) is the predicted probability for class (y), (n_y) is the number of training samples for class (y), and (\\beta) is exponential factor.

| PARAMETER     | DESCRIPTION                                                |
| ------------- | ---------------------------------------------------------- |
| `criterion`   | loss module **TYPE:** `Module`                             |
| `num_samples` | number of samples for each class **TYPE:** `Tensor`        |
| `beta`        | rebalancing exponent **TYPE:** `float` **DEFAULT:** `0.99` |

Source code in `holocron/nn/modules/loss.py`

```python
def __init__(self, criterion: nn.Module, num_samples: Tensor, beta: float = 0.99) -> None:
    super().__init__()
    self.criterion = criterion
    self.beta: float = beta
    cb_weights = (1 - beta) / (1 - beta**num_samples)
    if self.criterion.weight is None:
        self.criterion.weight: Tensor | None = cb_weights
    else:
        self.criterion.weight *= cb_weights.to(device=self.criterion.weight.device)  # ty: ignore[invalid-argument-type,possibly-missing-attribute]
```

## Convolution layers

### NormConv2d

```python
NormConv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros', eps: float = 1e-14)
```

Bases: `_NormConvNd`

Implements the normalized convolution module from ["Normalized Convolutional Neural Network"](https://arxiv.org/pdf/2005.05274v2.pdf).

In the simplest case, the output value of the layer with input size ((N, C\_{in}, H, W)) and output ((N, C\_{out}, H\_{out}, W\_{out})) can be precisely described as:

[ out(N_i, C\_{out_j}) = bias(C\_{out_j}) + \\sum\_{k = 0}^{C\_{in} - 1} weight(C\_{out_j}, k) \\star \\frac{input(N_i, k) - \\mu(N_i, k)}{\\sqrt{\\sigma^2(N_i, k) + \\epsilon}} ]

where: (\\star) is the valid 2D cross-correlation operator, (\\mu(N_i, k)) and (\\sigma²(N_i, k)) are the mean and variance of (input(N_i, k)) over all slices, (N) is a batch size, (C) denotes a number of channels, (H) is a height of input planes in pixels, and (W) is width in pixels.

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `in_channels`  | Number of channels in the input image **TYPE:** `int`                                                                     |
| `out_channels` | Number of channels produced by the convolution **TYPE:** `int`                                                            |
| `kernel_size`  | Size of the convolving kernel **TYPE:** `int`                                                                             |
| `stride`       | Stride of the convolution. **TYPE:** `int` **DEFAULT:** `1`                                                               |
| `padding`      | Zero-padding added to both sides of the input. **TYPE:** `int` **DEFAULT:** `0`                                           |
| `dilation`     | Spacing between kernel elements. **TYPE:** `int` **DEFAULT:** `1`                                                         |
| `groups`       | Number of blocked connections from input channels to output channels. **TYPE:** `int` **DEFAULT:** `1`                    |
| `bias`         | If True, adds a learnable bias to the output. **TYPE:** `bool` **DEFAULT:** `True`                                        |
| `padding_mode` | padding mode for the convolution. **TYPE:** `Literal['zeros', 'reflect', 'replicate', 'circular']` **DEFAULT:** `'zeros'` |
| `eps`          | a value added to the denominator for numerical stability. **TYPE:** `float` **DEFAULT:** `1e-14`                          |

Source code in `holocron/nn/modules/conv.py`

```python
def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    eps: float = 1e-14,
) -> None:
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super().__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        False,
        _pair(0),
        groups,
        bias,
        padding_mode,
        False,
        eps,
    )
```

### Add2d

```python
Add2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros', normalize_slices: bool = False, eps: float = 1e-14)
```

Bases: `_NormConvNd`

Implements the adder module from ["AdderNet: Do We Really Need Multiplications in Deep Learning?"](https://arxiv.org/pdf/1912.13200.pdf).

In the simplest case, the output value of the layer at position ((m, n)) in channel (c) with filter F of spatial size ((d, d)), intput size ((C\_{in}, H, W)) and output ((C\_{out}, H, W)) can be precisely described as:

[ out(m, n, c) = - \\sum\\limits\_{i=0}^d \\sum\\limits\_{j=0}^d \\sum\\limits\_{k=0}^{C\_{in}} |X(m + i, n + j, k) - F(i, j, k, c)| ]

where (C) denotes a number of channels, (H) is a height of input planes in pixels, and (W) is width in pixels.

| PARAMETER          | DESCRIPTION                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `in_channels`      | Number of channels in the input image **TYPE:** `int`                                                                     |
| `out_channels`     | Number of channels produced by the convolution **TYPE:** `int`                                                            |
| `kernel_size`      | Size of the convolving kernel **TYPE:** `int`                                                                             |
| `stride`           | Stride of the convolution. **TYPE:** `int` **DEFAULT:** `1`                                                               |
| `padding`          | Zero-padding added to both sides of the input. **TYPE:** `int` **DEFAULT:** `0`                                           |
| `dilation`         | Spacing between kernel elements. **TYPE:** `int` **DEFAULT:** `1`                                                         |
| `groups`           | Number of blocked connections from input channels to output channels. **TYPE:** `int` **DEFAULT:** `1`                    |
| `bias`             | If True, adds a learnable bias to the output. **TYPE:** `bool` **DEFAULT:** `True`                                        |
| `padding_mode`     | padding mode for the convolution. **TYPE:** `Literal['zeros', 'reflect', 'replicate', 'circular']` **DEFAULT:** `'zeros'` |
| `normalize_slices` | whether slices should be normalized before performing cross-correlation. **TYPE:** `bool` **DEFAULT:** `False`            |
| `eps`              | a value added to the denominator for numerical stability. **TYPE:** `float` **DEFAULT:** `1e-14`                          |

Source code in `holocron/nn/modules/conv.py`

```python
def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    normalize_slices: bool = False,
    eps: float = 1e-14,
) -> None:
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super().__init__(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        False,
        _pair(0),
        groups,
        bias,
        padding_mode,
        normalize_slices,
        eps,
    )
```

### SlimConv2d

```python
SlimConv2d(in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros', r: int = 32, L: int = 2)
```

Bases: `Module`

Implements the convolution module from ["SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping"](https://arxiv.org/pdf/2003.07469.pdf).

First, we compute channel-wise weights as follows:

[ z(c) = \\frac{1}{H \\cdot W} \\sum\\limits\_{i=1}^H \\sum\\limits\_{j=1}^W X\_{c,i,j} ]

where (X \\in \\mathbb{R}^{C \\times H \\times W}) is the input tensor, (H) is height in pixels, and (W) is width in pixels.

[ w = \\sigma(F\_{fc2}(\\delta(F\_{fc1}(z)))) ]

where (z \\in \\mathbb{R}^{C}) contains channel-wise statistics, (\\sigma) refers to the sigmoid function, (\\delta) refers to the ReLU function, (F\_{fc1}) is a convolution operation with kernel of size ((1, 1)) with (max(C/r, L)) output channels followed by batch normalization, and (F\_{fc2}) is a plain convolution operation with kernel of size ((1, 1)) with (C) output channels.

We then proceed with reconstructing and transforming both pathways:

[ X\_{top} = X \\odot w X\_{bot} = X \\odot \\check{w} ]

where (\\odot) refers to the element-wise multiplication and (\\check{w}) is the channel-wise reverse-flip of (w).

[ T\_{top} = F\_{top}(X\_{top}^{(1)} + X\_{top}^{(2)}) T\_{bot} = F\_{bot}(X\_{bot}^{(1)} + X\_{bot}^{(2)}) ]

where (X^{(1)}) and (X^{(2)}) are the channel-wise first and second halves of (X), (F\_{top}) is a convolution of kernel size ((3, 3)), and (F\_{bot}) is a convolution of kernel size ((1, 1)) reducing channels by half, followed by a convolution of kernel size ((3, 3)).

Finally we fuse both pathways to yield the output:

[ Y = T\_{top} \\oplus T\_{bot} ]

where (\\oplus) is the channel-wise concatenation.

| PARAMETER      | DESCRIPTION                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `in_channels`  | Number of channels in the input image **TYPE:** `int`                                                                     |
| `kernel_size`  | Size of the convolving kernel **TYPE:** `int`                                                                             |
| `stride`       | Stride of the convolution. **TYPE:** `int` **DEFAULT:** `1`                                                               |
| `padding`      | Zero-padding added to both sides of the input. **TYPE:** `int` **DEFAULT:** `0`                                           |
| `dilation`     | Spacing between kernel elements. **TYPE:** `int` **DEFAULT:** `1`                                                         |
| `groups`       | Number of blocked connections from input channels to output channels. **TYPE:** `int` **DEFAULT:** `1`                    |
| `bias`         | If True, adds a learnable bias to the output. **TYPE:** `bool` **DEFAULT:** `True`                                        |
| `padding_mode` | padding mode for the convolution. **TYPE:** `Literal['zeros', 'reflect', 'replicate', 'circular']` **DEFAULT:** `'zeros'` |
| `r`            | squeezing divider. **TYPE:** `int` **DEFAULT:** `32`                                                                      |
| `L`            | minimum squeezed channels. **TYPE:** `int` **DEFAULT:** `2`                                                               |

Source code in `holocron/nn/modules/conv.py`

```python
def __init__(
    self,
    in_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    r: int = 32,
    L: int = 2,  # noqa: N803
) -> None:
    super().__init__()
    self.fc1 = nn.Conv2d(in_channels, max(in_channels // r, L), 1)
    self.bn = nn.BatchNorm2d(max(in_channels // r, L))
    self.fc2 = nn.Conv2d(max(in_channels // r, L), in_channels, 1)
    self.conv_top = nn.Conv2d(
        in_channels // 2, in_channels // 2, kernel_size, stride, padding, dilation, groups, bias, padding_mode
    )
    self.conv_bot1 = nn.Conv2d(in_channels // 2, in_channels // 4, 1)
    self.conv_bot2 = nn.Conv2d(
        in_channels // 4, in_channels // 4, kernel_size, stride, padding, dilation, groups, bias, padding_mode
    )
```

### PyConv2d

```python
PyConv2d(in_channels: int, out_channels: int, kernel_size: int, num_levels: int = 2, padding: int = 0, groups: list[int] | None = None, **kwargs: Any)
```

Bases: `ModuleList`

Implements the convolution module from ["Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"](https://arxiv.org/pdf/2006.11538.pdf).

| PARAMETER      | DESCRIPTION                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------- |
| `in_channels`  | Number of channels in the input image **TYPE:** `int`                                       |
| `out_channels` | Number of channels produced by the convolution **TYPE:** `int`                              |
| `kernel_size`  | Size of the convolving kernel **TYPE:** `int`                                               |
| `num_levels`   | number of stacks in the pyramid. **TYPE:** `int` **DEFAULT:** `2`                           |
| `padding`      | Zero-padding added to both sides of the input. **TYPE:** `int` **DEFAULT:** `0`             |
| `groups`       | Number of blocked connections from input channels to output channels. **TYPE:** \`list[int] |
| `kwargs`       | keyword args of torch.nn.Conv2d. **TYPE:** `Any` **DEFAULT:** `{}`                          |

Source code in `holocron/nn/modules/conv.py`

```python
def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    num_levels: int = 2,
    padding: int = 0,
    groups: list[int] | None = None,
    **kwargs: Any,
) -> None:
    if num_levels == 1:
        super().__init__([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups[0] if isinstance(groups, list) else 1,
                **kwargs,
            )
        ])
    else:
        exp2 = int(math.log2(num_levels))
        reminder = num_levels - 2**exp2
        out_chans = [out_channels // 2 ** (exp2 + 1)] * (2 * reminder) + [out_channels // 2**exp2] * (
            num_levels - 2 * reminder
        )

        k_sizes = [kernel_size + 2 * idx for idx in range(num_levels)]
        if groups is None:
            groups = [1] + [
                min(2 ** (2 + idx), out_chan)
                for idx, out_chan in zip(range(num_levels - 1), out_chans[1:], strict=True)
            ]
        elif not isinstance(groups, list) or len(groups) != num_levels:
            raise ValueError("The argument `group` is expected to be a list of integer of size `num_levels`.")
        paddings = [padding + idx for idx in range(num_levels)]

        super().__init__([
            nn.Conv2d(in_channels, out_chan, k_size, padding=padding, groups=group, **kwargs)
            for out_chan, k_size, padding, group in zip(out_chans, k_sizes, paddings, groups, strict=True)
        ])
    self.num_levels: int = num_levels
```

### Involution2d

```python
Involution2d(in_channels: int, kernel_size: int, padding: int = 0, stride: int = 1, groups: int = 1, dilation: int = 1, reduction_ratio: float = 1)
```

Bases: `Module`

Implements the convolution module from ["Involution: Inverting the Inherence of Convolution for Visual Recognition"](https://arxiv.org/pdf/2103.06255.pdf), adapted from the proposed PyTorch implementation in the paper.

| PARAMETER         | DESCRIPTION                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| `in_channels`     | Number of channels in the input image **TYPE:** `int`                                                  |
| `kernel_size`     | Size of the convolving kernel **TYPE:** `int`                                                          |
| `padding`         | Zero-padding added to both sides of the input. **TYPE:** `int` **DEFAULT:** `0`                        |
| `stride`          | Stride of the convolution. **TYPE:** `int` **DEFAULT:** `1`                                            |
| `groups`          | Number of blocked connections from input channels to output channels. **TYPE:** `int` **DEFAULT:** `1` |
| `dilation`        | Spacing between kernel elements. **TYPE:** `int` **DEFAULT:** `1`                                      |
| `reduction_ratio` | reduction ratio of the channels to generate the kernel **TYPE:** `float` **DEFAULT:** `1`              |

Source code in `holocron/nn/modules/conv.py`

```python
def __init__(
    self,
    in_channels: int,
    kernel_size: int,
    padding: int = 0,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    reduction_ratio: float = 1,
) -> None:
    super().__init__()

    self.groups: int = groups
    self.k_size: int = kernel_size

    self.pool: nn.AvgPool2d | None = nn.AvgPool2d(stride, stride) if stride > 1 else None
    self.reduce = nn.Conv2d(in_channels, int(in_channels // reduction_ratio), 1)
    self.span = nn.Conv2d(int(in_channels // reduction_ratio), kernel_size**2 * groups, 1)
    self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
```

## Regularization layers

### DropBlock2d

```python
DropBlock2d(p: float = 0.1, block_size: int = 7, inplace: bool = False)
```

Bases: `Module`

Implements the DropBlock module from ["DropBlock: A regularization method for convolutional networks"](https://arxiv.org/pdf/1810.12890.pdf)

| PARAMETER    | DESCRIPTION                                                                                |
| ------------ | ------------------------------------------------------------------------------------------ |
| `p`          | probability of dropping activation value **TYPE:** `float` **DEFAULT:** `0.1`              |
| `block_size` | size of each block that is expended from the sampled mask **TYPE:** `int` **DEFAULT:** `7` |
| `inplace`    | whether the operation should be done inplace **TYPE:** `bool` **DEFAULT:** `False`         |

Source code in `holocron/nn/modules/dropblock.py`

```python
def __init__(self, p: float = 0.1, block_size: int = 7, inplace: bool = False) -> None:
    super().__init__()
    self.p: float = p
    self.block_size: int = block_size
    self.inplace: bool = inplace
```

## Downsampling

### ConcatDownsample2d

```python
ConcatDownsample2d(scale_factor: int)
```

Bases: `Module`

Implements a loss-less downsampling operation described in ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf) by stacking adjacent information on the channel dimension.

| PARAMETER      | DESCRIPTION                            |
| -------------- | -------------------------------------- |
| `scale_factor` | spatial scaling factor **TYPE:** `int` |

Source code in `holocron/nn/modules/downsample.py`

```python
def __init__(self, scale_factor: int) -> None:
    super().__init__()
    self.scale_factor: int = scale_factor
```

### GlobalAvgPool2d

```python
GlobalAvgPool2d(flatten: bool = False)
```

Bases: `Module`

Fast implementation of global average pooling from ["TResNet: High Performance GPU-Dedicated Architecture"](https://arxiv.org/pdf/2003.13630.pdf)

| PARAMETER | DESCRIPTION                                                                         |
| --------- | ----------------------------------------------------------------------------------- |
| `flatten` | whether spatial dimensions should be squeezed **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/nn/modules/downsample.py`

```python
def __init__(self, flatten: bool = False) -> None:
    super().__init__()
    self.flatten: bool = flatten
```

### GlobalMaxPool2d

```python
GlobalMaxPool2d(flatten: bool = False)
```

Bases: `Module`

Fast implementation of global max pooling from ["TResNet: High Performance GPU-Dedicated Architecture"](https://arxiv.org/pdf/2003.13630.pdf)

| PARAMETER | DESCRIPTION                                                                         |
| --------- | ----------------------------------------------------------------------------------- |
| `flatten` | whether spatial dimensions should be squeezed **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/nn/modules/downsample.py`

```python
def __init__(self, flatten: bool = False) -> None:
    super().__init__()
    self.flatten: bool = flatten
```

### BlurPool2d

```python
BlurPool2d(channels: int, kernel_size: int = 3, stride: int = 2)
```

Bases: `Module`

Ross Wightman's [implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/blur_pool.py) of blur pooling module as described in ["Making Convolutional Networks Shift-Invariant Again"](https://arxiv.org/pdf/1904.11486.pdf).

| PARAMETER     | DESCRIPTION                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `channels`    | Number of input channels **TYPE:** `int`                                                                  |
| `kernel_size` | binomial filter size for blurring. currently supports 3 (default) and 5. **TYPE:** `int` **DEFAULT:** `3` |
| `stride`      | downsampling filter stride **TYPE:** `int` **DEFAULT:** `2`                                               |

Source code in `holocron/nn/modules/downsample.py`

```python
def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2) -> None:
    super().__init__()
    self.channels: int = channels
    if kernel_size <= 1:
        raise AssertionError
    self.kernel_size: int = kernel_size
    self.stride: int = stride
    pad_size = [get_padding(kernel_size, stride, dilation=1)] * 4
    self.padding = nn.ReflectionPad2d(pad_size)  # type: ignore[arg-type]
    self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.kernel_size - 1)).coeffs)  # for torchscript compat
    self.kernel: dict[str, Tensor] = {}  # lazy init by device for DataParallel compat
```

### SPP

```python
SPP(kernel_sizes: list[int])
```

Bases: `ModuleList`

SPP layer from ["Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"](https://arxiv.org/pdf/1406.4729.pdf).

| PARAMETER      | DESCRIPTION                                        |
| -------------- | -------------------------------------------------- |
| `kernel_sizes` | kernel sizes of each pooling **TYPE:** `list[int]` |

Source code in `holocron/nn/modules/downsample.py`

```python
def __init__(self, kernel_sizes: list[int]) -> None:
    super().__init__([nn.MaxPool2d(k_size, stride=1, padding=k_size // 2) for k_size in kernel_sizes])
```

### ZPool

```python
ZPool(dim: int = 1)
```

Bases: `Module`

Z-pool layer from ["Rotate to Attend: Convolutional Triplet Attention Module"](https://arxiv.org/pdf/2010.03045.pdf).

| PARAMETER | DESCRIPTION                                               |
| --------- | --------------------------------------------------------- |
| `dim`     | dimension to pool across **TYPE:** `int` **DEFAULT:** `1` |

Source code in `holocron/nn/modules/downsample.py`

```python
def __init__(self, dim: int = 1) -> None:
    super().__init__()
    self.dim: int = dim
```

## Attention

### SAM

```python
SAM(in_channels: int)
```

Bases: `Module`

SAM layer from ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/pdf/1807.06521.pdf) modified in ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf).

| PARAMETER     | DESCRIPTION                    |
| ------------- | ------------------------------ |
| `in_channels` | input channels **TYPE:** `int` |

Source code in `holocron/nn/modules/attention.py`

```python
def __init__(self, in_channels: int) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, 1, 1)
```

### LambdaLayer

```python
LambdaLayer(in_channels: int, out_channels: int, dim_k: int, n: int | None = None, r: int | None = None, num_heads: int = 4, dim_u: int = 1)
```

Bases: `Module`

Lambda layer from ["LambdaNetworks: Modeling long-range interactions without attention"](https://openreview.net/pdf?id=xTJEN-ggl1b). The implementation was adapted from [lucidrains](https://github.com/lucidrains/lambda-networks/blob/main/lambda_networks/lambda_networks.py).

| PARAMETER      | DESCRIPTION                                                      |
| -------------- | ---------------------------------------------------------------- |
| `in_channels`  | input channels **TYPE:** `int`                                   |
| `out_channels` | output channels **TYPE:** `int`                                  |
| `dim_k`        | key dimension **TYPE:** `int`                                    |
| `n`            | number of input pixels **TYPE:** \`int                           |
| `r`            | receptive field for relative positional encoding **TYPE:** \`int |
| `num_heads`    | number of attention heads **TYPE:** `int` **DEFAULT:** `4`       |
| `dim_u`        | intra-depth dimension **TYPE:** `int` **DEFAULT:** `1`           |

Source code in `holocron/nn/modules/lambda_layer.py`

```python
def __init__(
    self,
    in_channels: int,
    out_channels: int,
    dim_k: int,
    n: int | None = None,
    r: int | None = None,
    num_heads: int = 4,
    dim_u: int = 1,
) -> None:
    super().__init__()
    self.u: int = dim_u
    self.num_heads: int = num_heads

    if out_channels % num_heads != 0:
        raise AssertionError("values dimension must be divisible by number of heads for multi-head query")
    dim_v = out_channels // num_heads

    # Project input and context to get queries, keys & values
    self.to_q = nn.Conv2d(in_channels, dim_k * num_heads, 1, bias=False)
    self.to_k = nn.Conv2d(in_channels, dim_k * dim_u, 1, bias=False)
    self.to_v = nn.Conv2d(in_channels, dim_v * dim_u, 1, bias=False)

    self.norm_q = nn.BatchNorm2d(dim_k * num_heads)
    self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

    self.local_contexts: bool = r is not None
    if r is not None:
        if r % 2 != 1:
            raise AssertionError("Receptive kernel size should be odd")
        self.padding: int = r // 2
        self.R = nn.Parameter(torch.randn(dim_k, dim_u, 1, r, r))
    else:
        if n is None:
            raise AssertionError("You must specify the total sequence length (h x w)")
        self.pos_emb = nn.Parameter(torch.randn(n, n, dim_k, dim_u))
```

### TripletAttention

```python
TripletAttention()
```

Bases: `Module`

Triplet attention layer from ["Rotate to Attend: Convolutional Triplet Attention Module"](https://arxiv.org/pdf/2010.03045.pdf). This implementation is based on the [one](https://github.com/LandskapeAI/triplet-attention/blob/master/MODELS/triplet_attention.py) from the paper's authors.

Source code in `holocron/nn/modules/attention.py`

```python
def __init__(self) -> None:
    super().__init__()
    self.c_branch = DimAttention(dim=1)
    self.h_branch = DimAttention(dim=2)
    self.w_branch = DimAttention(dim=3)
```
