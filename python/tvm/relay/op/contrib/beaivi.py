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
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay import transform
from tvm.relay.analysis import free_vars
from .register import register_pattern_table
from tvm.relay.op.contrib.register import get_pattern_table

logger = logging.getLogger("BEAIVI")

# %0 = nn.pad(%input_1, 83 /* ty=int32 span=functional_1/activation/Relu;functional_1/batch_normalization/FusedBatchNormV3;functional_1/conv2d/BiasAdd/ReadVariableOp/resource;functional_1/conv2d/BiasAdd;functional_1/conv2d_4/Conv2D;functional_1/conv2d/Conv2D1:0:0 */, pad_width=[[0, 0], [4, 5], [1, 1], [0, 0]]) /* ty=Tensor[(1, 58, 12, 1), int8] */;
#   %1 = nn.conv2d(%0, meta[relay.Constant][0] /* ty=Tensor[(10, 4, 1, 64), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[10, 4], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32") /* ty=Tensor[(1, 25, 5, 64), int32] */;
#   %2 = subtract(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 1, 1, 64), int32] */) /* ty=Tensor[(1, 25, 5, 64), int32] */;
#   %3 = nn.bias_add(%2, meta[relay.Constant][2] /* ty=Tensor[(64), int32] */, axis=3) /* ty=Tensor[(1, 25, 5, 64), int32] span=functional_1/activation/Relu;functional_1/batch_normalization/FusedBatchNormV3;functional_1/conv2d/BiasAdd/ReadVariableOp/resource;functional_1/conv2d/BiasAdd;functional_1/conv2d_4/Conv2D;functional_1/conv2d/Conv2D1:0:0 */;
#   %4 = fixed_point_multiply_per_axis(%3, meta[relay.Constant][3] /* ty=Tensor[(64), int32] */, meta[relay.Constant][4] /* ty=Tensor[(64), int32] */, meta[relay.Constant][5] /* ty=Tensor[(64), int32] */, is_rshift_required=True, axes=[3]) /* ty=Tensor[(1, 25, 5, 64), int32] */;
#   %5 = add(-128 /* ty=int32 span=functional_1/activation/Relu;functional_1/batch_normalization/FusedBatchNormV3;functional_1/conv2d/BiasAdd/ReadVariableOp/resource;functional_1/conv2d/BiasAdd;functional_1/conv2d_4/Conv2D;functional_1/conv2d/Conv2D1:0:0 */, %4) /* ty=Tensor[(1, 25, 5, 64), int32] */;
#   %6 = clip(%5, a_min=-128f, a_max=127f) /* ty=Tensor[(1, 25, 5, 64), int32] */;
#   %7 = cast(%6, dtype="int8") /* ty=Tensor[(1, 25, 5, 64), int8] */;


def conv2d_pattern(with_pad):
    pattern_input = wildcard()
    weights = wildcard()
    bias = wildcard()
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
    pattern = is_op("cast")(pattern)
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
    # NOTE: Order is import here, since patterns are checked for in order 0->N.
    # Padded are less inclusive so those need to be checked first
    return [conv2d_padded, conv2d, conv2d_depthwise_padded, conv2d_depthwise]


class LegalizeQnnOpForBeaivi(DFPatternCallback):
    def __init__(self):
        super(LegalizeQnnOpForBeaivi, self).__init__()

        # Define common wildcards
        self.pattern_input = wildcard()
        self.wgh = wildcard()
        self.bias = wildcard()

        # Define pad operation (if it exists)
        self.pad = is_op("nn.pad")(self.pattern_input)

        # Define conv2d operation, which can take either the original src or a padded src
        self.conv = is_op("nn.conv2d")((self.pad | self.pattern_input), self.wgh)

        # Rest of the pattern
        self.sub = is_op("subtract")(self.conv, is_constant())
        self.bias_add = is_op("nn.bias_add")(self.sub, self.bias)
        self.fixed_point = is_op("fixed_point_multiply_per_axis")(
            self.bias_add, is_constant(), is_constant(), is_constant()
        )
        self.add = is_op("add")(self.fixed_point, is_constant())
        self.clip = is_op("clip")(self.add)
        self.cast = is_op("cast")(self.clip)

        # Full pattern
        self.pattern = self.cast

    def callback(self, pre, post, node_map):
        root = node_map[self.root][0]
        output = relay.Call(conv2d_node.op, conv2d_node.args, attrs=new_attrs)

        return output


def legalize_qnn_for_beaivi(mod):
    print("Legalizing qnn for Beaivi")

    desired_layouts = {"qnn.conv2d": ["NHWC", "OHWI"]}

    beaivi_patterns = get_pattern_table("beaivi")

    # Pre process
    preprocessor_pass = tvm.transform.Sequential(
        [
            relay.qnn.transform.Legalize(),
            relay.qnn.transform.CanonicalizeOps(),
            relay.transform.InferType(),
            # relay.transform.ConvertLayout(desired_layouts),
            relay.transform.SimplifyExpr(),
            transform.FoldConstant(),
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
        mod = preprocessor_pass(mod)
        print(mod)
        # mod["main"] = rewrite(LegalizeQnnOpForBeaivi(), mod["main"])
        mod = annotation_pass(mod)
    return mod
