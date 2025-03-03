# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging
import tvm.ir
from tvm import relay
from ...dataflow_pattern import DFPatternCallback, is_constant, is_expr, is_op, rewrite, wildcard
from tvm.relay.expr import Call, GlobalVar, TupleGetItem, const
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm.relay import transform
from tvm.relay.analysis import free_vars
from .register import register_pattern_table
from tvm.relay.op.contrib.register import get_pattern_table
from ..strategy.generic import is_depthwise_conv2d
from tvm.relay.qnn.op import requantize

import numpy as np


logger = logging.getLogger("BEAIVI")


def conv2d_pattern(with_pad):
    pattern_input = wildcard()
    weights = is_constant()
    bias = is_constant()
    if with_pad:
        pattern = is_op("nn.pad")(pattern_input, is_constant())
        pattern = is_op("nn.conv2d")(pattern, weights)
    else:
        pattern = is_op("nn.conv2d")(pattern_input, weights)
    pattern = is_op("subtract")(pattern, is_constant())
    pattern = is_op("nn.bias_add")(pattern, bias)
    # Scale factor, shift, zero_point
    pattern = is_op("fixed_point_multiply_per_axis")(
        pattern, is_constant(), is_constant(), is_constant()
    )

    pattern = is_op("add")(pattern, is_constant())
    pattern = is_op("clip")(pattern)
    # Optional cast for certain fuse patterns
    pattern = pattern | is_op("cast")(pattern)
    return pattern


def dense1d_pattern():
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    pattern = is_op("nn.dense")(data, weight)
    pattern = is_op("subtract")(pattern, is_constant())
    pattern = is_op("nn.bias_add")(pattern, bias)
    pattern = is_op("fixed_point_multiply")(pattern, is_constant(), is_constant())
    pattern = is_op("add")(pattern, is_constant())
    pattern = is_op("clip")(pattern)


def qnn_conv2d_pattern():
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    pattern = is_op("qnn.conv2d")(
        data, weight, is_constant(), is_constant(), is_constant(), is_constant()
    )
    pattern = is_op("nn.bias_add")(pattern, bias)
    pattern = is_op("qnn.requantize")
    pattern = is_op("clip")
    return pattern


def qnn_avg_pool2d_pattern():
    data = wildcard()
    pattern = is_op("nn.avg_pool2d")(data)
    pattern = is_op("cast")(pattern)
    pattern = is_op("reshape")(pattern)
    return pattern


def pattern_name(pattern):
    return str(pattern.op.name)


def check_is_depthwise(pattern):
    print("Depthwise")
    # Find conv
    conv = None
    if pattern_name(pattern) == "cast":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "clip":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "add":
        pattern = pattern.args[1]
    if pattern_name(pattern) == "fixed_point_multiply_per_axis":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "nn.bias_add":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "subtract":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "nn.conv2d":
        conv = pattern
    if not conv:
        raise Exception(f"Unknown pattern name {str(pattern.op.name)}")
    return conv.attrs.groups > 1


def check_is_not_depthwise(pattern):
    print("Not Depthwise")
    conv = None
    if pattern_name(pattern) == "cast":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "clip":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "add":
        pattern = pattern.args[1]
    if pattern_name(pattern) == "fixed_point_multiply_per_axis":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "nn.bias_add":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "subtract":
        pattern = pattern.args[0]
    if pattern_name(pattern) == "nn.conv2d":
        conv = pattern
    if not conv:
        raise Exception(f"Unknown pattern name {str(pattern.op.name)}")
    return conv.attrs.groups == 1


@register_pattern_table("beaivi")
def pattern_table():
    print("Patterns")
    conv2d = (
        "beaivi.conv2d",
        conv2d_pattern(False),
        check_is_not_depthwise,
    )
    conv2d_padded = (
        "beaivi.conv2d",
        conv2d_pattern(True),
        check_is_not_depthwise,
    )
    conv2d_depthwise = (
        "beaivi.conv2d_depthwise",
        conv2d_pattern(False),
        check_is_depthwise,
    )
    conv2d_depthwise_padded = (
        "beaivi.conv2d_depthwise",
        conv2d_pattern(True),
        check_is_depthwise,
    )
    avg_pool2d_pat = ("beaivi.avg_pool2d", qnn_avg_pool2d_pattern())
    dense1d_pat = ("beaivi.dense1d", dense1d_pattern())

    # qnn_conv2d = ("beaivi.conv2d", qnn_conv2d_pattern())
    # NOTE: Order is import here, since patterns are checked for in order 0->N.
    # Padded are less inclusive so those need to be checked first
    return [
        conv2d_padded,
        conv2d,
        conv2d_depthwise_padded,
        conv2d_depthwise,
        avg_pool2d_pat,
        # dense1d_pat,
    ]
    # return [qnn_conv2d]


class LegalizeQnnOpForBeaivi(DFPatternCallback):
    def __init__(self):
        super(LegalizeQnnOpForBeaivi, self).__init__()
        # Define the pattern to match
        self.src = wildcard()
        self.weights = wildcard()
        self.bias = wildcard()
        self.root = is_op("qnn.conv2d")(
            self.src, self.weights, is_constant(), is_constant(), is_constant(), is_constant()
        )
        self.bias_add = is_op("nn.bias_add")(self.root, self.bias)
        self.requantize = is_op("qnn.requantize")(
            self.bias_add, wildcard(), wildcard(), wildcard(), wildcard()
        )
        self.clip = is_op("clip")(self.requantize) | self.requantize  # Optional clip

        # The final pattern to match
        self.pattern = self.clip

    def callback(self, pre, post, node_map):
        """Rewrite qnn.conv2d + bias_add + requantize into a fixed-point implementation."""
        root = node_map[self.root][0]
        bias = node_map.get(self.bias, default=[relay.const(0, dtype="int32")])[0]
        src = node_map[self.src][0]
        weights = node_map[self.weights][0]

        requantize = node_map[self.requantize][0]

        # Extract requantization parameters
        (
            rq_input_scale,
            rq_input_zero_point,
            rq_output_scale,
            rq_output_zero_point,
        ) = requantize.args[1:5]
        input_zero_point, kernel_zero_point, input_scale, kernel_scale = root.args[2:6]

        input_scale = input_scale.data.numpy()
        kernel_scale = kernel_scale.data.numpy()
        rq_input_scale = rq_input_scale.data.numpy()
        rq_input_zero_point = rq_input_zero_point.data.numpy()
        rq_output_scale = rq_output_scale.data.numpy()
        rq_output_zero_point = rq_output_zero_point.data.numpy()

        print("Input scale:", input_scale)
        print("Input zp:", input_zero_point.data.numpy())
        print("Kernel scale:", kernel_scale)
        print("Kernel zp:", kernel_zero_point.data.numpy())
        print("Rq Input scale:", rq_input_scale)
        print("Rq Input zp:", rq_input_zero_point)
        print("Rq Output scale:", rq_output_scale)
        print("Rq Output zp:", rq_output_zero_point)

        scales, shifts = convert_to_fixed_point(rq_input_scale, rq_output_scale)

        output = tvm.relay.Call(
            root.op,
            [
                src,
                weights,
                input_zero_point,
                kernel_zero_point,
                relay.const(1.0, dtype="float32"),  # Scales are handeld manually
                relay.const(1.0, dtype="float32"),
            ],
            root.attrs,
            root.type_args,
            root.span,
        )

        output = output + bias
        output = output * scales
        output = tvm.relay.fixed_point_multiply(output, scales, shifts)

        output = relay.cast(output, "int8")

        return output


def convert_to_fixed_point(input_scale, output_scale):
    input_scale = np.array(input_scale, dtype=np.float64)
    output_scale = np.array(output_scale, dtype=np.float64)
    scale_ratio = np.array(output_scale / input_scale, dtype=np.float64)
    shifts = []
    scales = []
    for x in scale_ratio:
        scales.append(get_fixed_point_multiplier_shift(x)[0])
        shifts.append(get_fixed_point_multiplier_shift(x)[1])
    print("Fixed_point_scale:", scales)
    print("Fixed_point_shift:", shifts)
    return scales, shifts


# NOTE: This works, don't change!
# def get_fixed_point_multiplier_shift(double_multiplier):
#     if double_multiplier == 0.0:
#         return 0, 0

#     significand, exponent = np.frexp(double_multiplier)
#     significand = np.int32(np.round(significand * (1 << 31)))

#     return significand, exponent - 1  # Adjust exponent to reflect the shift


def get_fixed_point_multiplier_shift(double_multiplier):
    if double_multiplier == 0.0:
        return 0, 0

    # Get the significand and exponent using frexp
    significand_d, exponent = np.frexp(double_multiplier)

    # Convert the significand to an integer representation
    # Multiply by 2^31 and round to the nearest integer
    significand_int64 = np.floor(significand_d * (1 << 31))

    # Ensure the significand fits within 31 bits
    if significand_int64 == (1 << 31):
        significand_int64 //= 2
        exponent += 1

    # Ensure the significand fits within the range of a 32-bit integer
    if significand_int64 > np.iinfo(np.int32).max:
        raise ValueError("Significand exceeds the range of a 32-bit integer.")

    significand = np.int32(significand_int64)

    # Calculate the shift
    shift = exponent

    return significand, shift


def legalize_qnn_for_beaivi(mod):
    print("Legalizing qnn for Beaivi")

    # desired_layouts = {"nn.conv2d": ["NHWC", "HWOI"]}
    desired_layouts = {"qnn.conv2d": ["NHWC", "HWIO"]}

    beaivi_patterns = get_pattern_table("beaivi")

    # Pre process
    preprocessor_pass = tvm.transform.Sequential(
        [
            # relay.transform.ConvertLayout(desired_layouts),
            relay.qnn.transform.CanonicalizeOps(),
            relay.qnn.transform.Legalize(),
            relay.transform.InferType(),
            relay.transform.SimplifyExpr(),
            transform.FoldScaleAxis(),
        ]
    )

    annotation_pass = tvm.transform.Sequential(
        [
            relay.transform.MergeComposite(beaivi_patterns),
            relay.transform.AnnotateTarget(["beaivi"]),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(),
            relay.transform.FoldConstant(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        # mod = convert_layout_module(mod)
        mod = preprocessor_pass(mod)
        # mod["main"] = rewrite(LegalizeQnnOpForBeaivi(), mod["main"])
        mod = annotation_pass(mod)
        print(mod)
    return mod
