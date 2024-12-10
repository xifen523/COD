# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, ResNetLayer)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

# __all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
#            'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
#            'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
#            'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
#            'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
#            'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'ResNetLayer')

__all__ = {
            'Conv' : Conv,
            'Conv2' : Conv2,
            'LightConv' : LightConv,
            'RepConv' : RepConv,
            'DWConv' : DWConv,
            'DWConvTranspose2d' : DWConvTranspose2d,
            'ConvTranspose' : ConvTranspose,
            'Focus' : Focus,
            'GhostConv' : GhostConv,
            'ChannelAttention' : ChannelAttention,
            'SpatialAttention' : SpatialAttention,
            'CBAM' : CBAM,
            'Concat' : Concat,
            'TransformerLayer' : TransformerLayer,
            'TransformerBlock' : TransformerBlock,
            'MLPBlock' : MLPBlock,
            'LayerNorm2d' : LayerNorm2d,
            'DFL' : DFL,
            'HGBlock' : HGBlock,
            'HGStem' : HGStem,
            'SPP' : SPP,
            'SPPF' : SPPF,
            'C1' : C1,
            'C2' : C2,
            'C3' : C3,
            'C2f' : C2f,
            'C3x' : C3x,
            'C3TR' : C3TR,
            'C3Ghost' : C3Ghost,
            'GhostBottleneck' : GhostBottleneck,
            'Bottleneck' : Bottleneck,
            'BottleneckCSP' : BottleneckCSP,
            'Proto' : Proto,
            'Detect' : Detect,
            'Segment' : Segment,
            'Pose' : Pose,
            'Classify' : Classify,
            'TransformerEncoderLayer' : TransformerEncoderLayer,
            'RepC3' : RepC3,
            'RTDETRDecoder' : RTDETRDecoder,
            'AIFI' : AIFI,
            'DeformableTransformerDecoder' : DeformableTransformerDecoder,
            'DeformableTransformerDecoderLayer' : DeformableTransformerDecoderLayer,
            'MSDeformAttn' : MSDeformAttn,
            'MLP' : MLP,
            'ResNetLayer' : ResNetLayer,
        }