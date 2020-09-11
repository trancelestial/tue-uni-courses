#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import os
import torch
# import center_surround_cuda as csc
from torch.utils.cpp_extension import load

# TODO: use the Just In Time compiler from pytorch to load the module that you
# exported from c++.
dirname = os.path.dirname(os.path.relpath(__file__))
sources = [os.path.join(dirname, f) for f in
        ["center_surround_convolution.cu"]]
my_center_surround_convolution = load(name="my_center_surround_convolution",
        sources=sources,
        verbose=True)


# e) Load your the exported python module in center surround convolution.py and
# implement the torch.autograd.Function class center surround convolution.
class center_surround_convolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                I: torch.Tensor,
                w_c: torch.Tensor,
                w_s: torch.Tensor,
                w_b: torch.Tensor) -> torch.Tensor:
        outputs = my_center_surround_convolution.forward(I, w_c, w_s, w_b)
        ctx.save_for_backward(I, w_c, w_s, w_b)
        return outputs[0]

    @staticmethod
    def backward(ctx, dL_dO: torch.Tensor) -> Tuple[torch.Tensor]:
        I, w_c, w_s, w_b = ctx.saved_tensors
        O_dL_dI, O_dL_dw_c, O_dL_dw_s, O_dL_dw_b = my_center_surround_convolution.backward(dL_dO, I, w_c, w_s, w_b)

        return O_dL_dI, O_dL_dw_c, O_dL_dw_s, O_dL_dw_b


# f) In the same file create a new torch.nn.Module called
# CenterSurroundConvolution which can be used as layer in a neural network.
# TODO: Create the CenterSurroundConvolution Module that represents a layer.
