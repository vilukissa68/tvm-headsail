/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ir/transform.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

#include "../../../transforms/pattern_utils.h"
#include "../../utils.h"
#include "./codegen_beaivi.h"
// #include "../codegen_c/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

// Struct to hold all arguments that are to be passed to the function call to the layer
struct Callables {
  std::string layer;
  std::vector<std::string> input;  // Comes from previous node
  std::vector<std::string> conv2d_attrs;
  std::vector<std::string> bias;
  std::vector<std::string> weights;
  std::vector<std::string> scales;
  std::vector<std::string> shifts;
  std::vector<std::string> zp_subtract;
  std::vector<std::string> zp;
  std::vector<std::string> input_sub_attr;
  std::vector<std::string> padding;
  std::vector<std::string> pool_attrs;
  std::vector<std::string> dense_attrs;
};

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

class CodegenBeaivi : public MemoizedExprTranslator<std::vector<Output>>,
                      public BeaiviCodegenCBase {
 public:
  // CodegenBeaivi(const std::string& id) { this->ext_func_id_ = id; }
  CodegenBeaivi(std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant,
                Array<String>* const_names, std::string ext_func_id)
      : const_name_to_constant_(const_name_to_constant),
        const_names_(const_names),
        ext_func_id_(std::move(ext_func_id)) {}

  Callables Conv2d_bias(const FunctionNode* callee, const CallNode* caller, bool depthwise) {
    Callables callables;
    const Conv2DAttrs* conv2d_attr = nullptr;
    const PadAttrs* padding_attrs = nullptr;
    const CallNode* conv2d_node = nullptr;

    callables.layer = "conv2d";
    // Increment total calls in module
    ++funcs_in_mod_;

    // Initalize padding
    int pad_top = 0;
    int pad_right = 0;
    int pad_left = 0;
    int pad_bottom = 0;

    // Initialize zero-point
    int zp = 0;

    // Extract function inputs from previous node
    callables.input.push_back(VisitExpr(caller->args[0])[0].name);  // Get inputs
    // callables.input_args.push_back(ExtractConstant(caller->args[2].as<ConstantNode>(), "bias"));

    const auto* current_call = callee->body.as<CallNode>();

    if (backend::IsOp(current_call, "cast")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "clip")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add")) {
      const int* zp_ptr =
          static_cast<const int*>((current_call->args[0].as<ConstantNode>()->data)->data);
      zp = zp_ptr[0];
      callables.zp_subtract.push_back(std::to_string(zp));
      current_call = current_call->args[1].as<CallNode>();
    }

    if (backend::IsOp(current_call, "fixed_point_multiply_per_axis")) {
      // callables.requantize_args = GetArgumentNames(current_call);
      std::string scale = ExtractConstant(current_call->args[1].as<ConstantNode>(), "scale");
      // std::string zp = ExtractConstant(current_call->args[2].as<ConstantNode>(), "zp");
      std::string shift = ExtractConstant(current_call->args[3].as<ConstantNode>(), "shift");
      callables.scales.push_back(scale);
      callables.shifts.push_back(shift);
      // callables.zp.push_back(zp);
      current_call = current_call->args[0].as<CallNode>();
    }

    if (backend::IsOp(current_call, "nn.bias_add")) {
      callables.bias.push_back(ExtractConstant(current_call->args[1].as<ConstantNode>(), "bias"));
      current_call = current_call->args[0].as<CallNode>();
    }

    if (backend::IsOp(current_call, "subtract")) {
      current_call = current_call->args[0].as<CallNode>();
    }

    if (backend::IsOp(current_call, "nn.conv2d")) {
      callables.weights.push_back(
          ExtractConstant(current_call->args[1].as<ConstantNode>(), "weight"));

      conv2d_attr = current_call->attrs.as<Conv2DAttrs>();
      ICHECK(conv2d_attr);

      // Save conv node for padding resolution
      conv2d_node = current_call;
      if (current_call->args[0].as<CallNode>()) {
        current_call = current_call->args[0].as<CallNode>();
      } else {
        callables.input_sub_attr.push_back(std::to_string(zp));
      }
    }

    if (backend::IsOp(current_call, "nn.pad")) {
      padding_attrs = current_call->attrs.as<PadAttrs>();
      pad_top = padding_attrs->pad_width[1][0].IntValue();
      pad_right = padding_attrs->pad_width[2][1].IntValue();
      pad_left = padding_attrs->pad_width[2][0].IntValue();
      pad_bottom = padding_attrs->pad_width[1][1].IntValue();
      std::cout << "Pad top " << pad_top << std::endl;
      std::cout << "Pad right " << pad_right << std::endl;
      std::cout << "Pad left " << pad_left << std::endl;
      std::cout << "Pad bottom " << pad_bottom << std::endl;

      const int* input_sub_value =
          static_cast<const int*>((current_call->args[1].as<ConstantNode>()->data)->data);
      // NOTE:(20250226 vaino-waltteri.granat@tuni.fi) When using TFLite quantized models we need to
      // corret zp by -128. This information can't be parsed from nn.pad so it's added here.
      callables.input_sub_attr.push_back(std::to_string(input_sub_value[0]));
    }

    auto ishape = GetShape(conv2d_node->args[0]->checked_type());  // Input shape
    auto wshape = GetShape(conv2d_node->args[1]->checked_type());  // Kernel shape

    int input_channels = ishape[3];
    int input_height = ishape[1] - pad_bottom - pad_top;
    int input_width = ishape[2] - pad_right - pad_left;

    callables.conv2d_attrs.push_back(std::to_string(input_height));    // Input height
    callables.conv2d_attrs.push_back(std::to_string(input_width));     // Input width
    callables.conv2d_attrs.push_back(std::to_string(input_channels));  // Input channels

    // NOTE: There is a discrepency with TFLite and TVM. TVM considers the kernel layout
    // to be OIHW, but looking at the weights they are HWIO
    callables.conv2d_attrs.push_back(std::to_string(wshape[0]));  // Kernels height
    callables.conv2d_attrs.push_back(std::to_string(wshape[1]));  // Kernels width
    callables.conv2d_attrs.push_back(std::to_string(wshape[3]));  // Kernels amount

    // Padding
    // int pad_top = conv2d_attr->padding[0].as<IntImmNode>()->value;
    // int pad_right = conv2d_attr->padding[3].as<IntImmNode>()->value;
    // int pad_left = conv2d_attr->padding[1].as<IntImmNode>()->value;
    // int pad_bottom = conv2d_attr->padding[2].as<IntImmNode>()->value;
    callables.conv2d_attrs.push_back(std::to_string(pad_top));     // Pad top
    callables.conv2d_attrs.push_back(std::to_string(pad_bottom));  // Pad bottom
    callables.conv2d_attrs.push_back(std::to_string(pad_left));    // Pad left
    callables.conv2d_attrs.push_back(std::to_string(pad_right));   // Pad right

    // Required work_buffer_size
    int required_work_buffer = (pad_top + input_height + pad_bottom) *
                               (pad_left + input_width + pad_right) * input_channels;
    work_buffers_.push_back(required_work_buffer);

    callables.conv2d_attrs.push_back(
        std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));  // Stride x
    callables.conv2d_attrs.push_back(
        std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));  // Stride y

    // We are in the end of the pattern, extract input subtraction
    return callables;
  }

  Callables Dense_1d(const FunctionNode* callee, const CallNode* caller, bool depthwise) {
    Callables callables;
    const DenseAttrs* dense_attrs = nullptr;
    std::cout << "Here" << std::endl;
    ++funcs_in_mod_;
    callables.layer = "dense1d";

    int zp = 0;

    // Extract function inputs from previous node
    callables.input.push_back(VisitExpr(caller->args[0])[0].name);  // Get inputs
    const auto* current_call = callee->body.as<CallNode>();

    if (backend::IsOp(current_call, "clip")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add")) {
      const int* zp_ptr =
          static_cast<const int*>((current_call->args[0].as<ConstantNode>()->data)->data);
      zp = zp_ptr[0];
      current_call = current_call->args[1].as<CallNode>();
    }
    if (backend::IsOp(current_call, "fixed_point_multiply")) {
      const FixedPointMultiplyAttrs* fp_attrs = current_call->attrs.as<FixedPointMultiplyAttrs>();
      int scale = fp_attrs->multiplier;
      int shift = fp_attrs->shift;
      callables.scales.push_back(std::to_string(scale));
      callables.shifts.push_back(std::to_string(shift));

      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.bias_add")) {
      callables.bias.push_back(ExtractConstant(current_call->args[1].as<ConstantNode>(), "bias"));
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "subtract")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.dense")) {
      // Input features
      dense_attrs = current_call->attrs.as<DenseAttrs>();

      callables.weights.push_back(
          ExtractConstant(current_call->args[1].as<ConstantNode>(), "weight"));

      auto ishape = GetShape(current_call->args[0]->checked_type());  // Input shape
      int input_channels = ishape[1];

      int out_features = dense_attrs->units.as<IntImmNode>()->value;
      std::cout << "out features: " << out_features << std::endl;

      callables.dense_attrs.push_back(std::to_string(input_channels));
      callables.dense_attrs.push_back(std::to_string(out_features));
      callables.input_sub_attr.push_back(std::to_string(zp));
    }

    return callables;
  }

  Callables Avg_pool2d(const FunctionNode* callee, const CallNode* caller, bool depthwise) {
    Callables callables;
    const AvgPool2DAttrs* pool_attrs = nullptr;

    ++funcs_in_mod_;
    callables.layer = "avgpool2d";

    // Extract function inputs from previous node
    callables.input.push_back(VisitExpr(caller->args[0])[0].name);  // Get inputs

    const auto* current_call = callee->body.as<CallNode>();

    if (backend::IsOp(current_call, "reshape")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "cast")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.avg_pool2d")) {
      // current_call = current_call->args[0].as<CallNode>();

      pool_attrs = current_call->attrs.as<AvgPool2DAttrs>();
      int stride_y = pool_attrs->strides[0].as<IntImmNode>()->value;
      int stride_x = pool_attrs->strides[1].as<IntImmNode>()->value;
      std::cout << "Stride Y: " << std::to_string(stride_y) << std::endl;
      std::cout << "Stride X: " << std::to_string(stride_x) << std::endl;
      int pool_size_y = pool_attrs->pool_size[0].as<IntImmNode>()->value;
      int pool_size_x = pool_attrs->pool_size[1].as<IntImmNode>()->value;
      std::cout << "Pool size Y: " << std::to_string(pool_size_y) << std::endl;
      std::cout << "Pool size X: " << std::to_string(pool_size_x) << std::endl;

      int pad_top = pool_attrs->padding[0].as<IntImmNode>()->value;
      int pad_right = pool_attrs->padding[3].as<IntImmNode>()->value;
      int pad_left = pool_attrs->padding[1].as<IntImmNode>()->value;
      int pad_bottom = pool_attrs->padding[2].as<IntImmNode>()->value;

      // Pattern input properties
      auto ishape = GetShape(current_call->args[0]->checked_type());  // Input shape
      int input_channels = ishape[3];
      int input_height = ishape[1] - pad_bottom - pad_top;
      int input_width = ishape[2] - pad_right - pad_left;

      callables.conv2d_attrs.push_back(std::to_string(input_height));
      callables.conv2d_attrs.push_back(std::to_string(input_width));
      callables.conv2d_attrs.push_back(std::to_string(input_channels));

      callables.pool_attrs.push_back(std::to_string(pool_size_y));
      callables.pool_attrs.push_back(std::to_string(pool_size_x));
      // callables.pool_attrs.push_back(std::to_string(stride_y));
      // callables.pool_attrs.push_back(std::to_string(stride_x));
      // NOTE: For now let's keep stride as 1
      callables.pool_attrs.push_back(std::to_string(1));
      callables.pool_attrs.push_back(std::to_string(1));

      // callables.pool_attrs.push_back(std::to_string(pad_top));
      // callables.pool_attrs.push_back(std::to_string(pad_right));
      // callables.pool_attrs.push_back(std::to_string(pad_left));
      // callables.pool_attrs.push_back(std::to_string(pad_bottom));
    }

    return callables;
  }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Beaivi codegen doesn't support: " << op->GetTypeKey();
  }
  // Generates function parameter
  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    std::cout << "Input variable:" << output.name << std::endl;
    return {output};
  }

  std::string ExtractConstant(const ConstantNode* cn, std::string name) {
    Output output;

    size_t const_id = const_name_to_constant_->size();

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    output.dtype = dtype;

    std::string const_var_name = CreateConstVar(ext_func_id_, const_id, name);
    output.name = const_var_name;

    std::vector<std::string> constant_values;

    tvm::runtime::NDArray data = cn->data;

    int ndim = data->ndim;
    int num_elements = 1;
    for (int i = 0; i < ndim; i++) {
      num_elements *= data->shape[i];
    }

    // Extract constant values
    // Convert the constant values to string and push to vector
    if (data->dtype.code == kDLFloat && data->dtype.bits == 32) {
      const float* values = static_cast<const float*>(data->data);
      for (int64_t i = 0; i < num_elements; ++i) {
        constant_values.push_back(std::to_string(values[i]));
      }
    } else if (data->dtype.code == kDLInt && data->dtype.bits == 32) {
      const int* values = static_cast<const int*>(data->data);
      for (int64_t i = 0; i < num_elements; ++i) {
        constant_values.push_back(std::to_string(values[i]));
      }
    } else if (data->dtype.code == kDLInt && data->dtype.bits == 8) {
      const int8_t* values = static_cast<const int8_t*>(data->data);
      for (int64_t i = 0; i < num_elements; ++i) {
        constant_values.push_back(std::to_string(values[i]));
      }
    }

    ExtractedConstArray extracted;
    extracted.arr = constant_values;
    extracted.dtype = dtype;
    extracted.size = num_elements;

    extracted_constants.insert({const_var_name, extracted});
    const_name_to_constant_->emplace(const_var_name, cn->data);
    const_names_->push_back(const_var_name);

    return output.name;
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
    Output output;

    size_t const_id = const_name_to_constant_->size();

    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    output.dtype = dtype;

    std::string const_var_name = CreateConstVar(ext_func_id_, const_id);
    output.name = const_var_name;

    std::vector<std::string> constant_values;

    tvm::runtime::NDArray data = cn->data;

    int ndim = data->ndim;
    int num_elements = 1;
    for (int i = 0; i < ndim; i++) {
      num_elements *= data->shape[i];
    }

    // Convert the constant values to string and push to vector
    if (data->dtype.code == kDLFloat && data->dtype.bits == 32) {
      const float* values = static_cast<const float*>(data->data);
      for (int64_t i = 0; i < num_elements; ++i) {
        constant_values.push_back(std::to_string(values[i]));
      }
    } else if (data->dtype.code == kDLInt && data->dtype.bits == 32) {
      const int* values = static_cast<const int*>(data->data);
      for (int64_t i = 0; i < num_elements; ++i) {
        constant_values.push_back(std::to_string(values[i]));
      }
    } else if (data->dtype.code == kDLInt && data->dtype.bits == 8) {
      const int8_t* values = static_cast<const int8_t*>(data->data);
      for (int64_t i = 0; i < num_elements; ++i) {
        constant_values.push_back(std::to_string(values[i]));
      }
    }

    ExtractedConstArray extracted;
    extracted.arr = constant_values;
    extracted.dtype = dtype;
    extracted.size = num_elements;

    extracted_constants.insert({const_var_name, extracted});
    const_name_to_constant_->emplace(const_var_name, cn->data);
    const_names_->push_back(const_var_name);

    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    GenerateBodyOutput ret;
    if (const auto* func = call->op.as<FunctionNode>()) {
      ret = GenerateCompositeFunctionCall(func, call);
    }

    ext_func_body_.push_back(ret.decl);
    return ret.outputs;
  }

  std::string JIT(const std::vector<Output>& out) {
    int work_buf_max = *std::max_element(work_buffers_.begin(), work_buffers_.end());
    for (std::vector<int>::iterator it = work_buffers_.begin(); it != work_buffers_.end(); ++it) {
      std::cout << "\t" << *it;
    }
    std::cout << std::endl;

    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_,
                   extracted_constants, work_buf_max, out);
  }

 private:
  // TODO: Fix this to parse composite
  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      std::cout << "I:" << i << std::endl;
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  std::vector<std::string> GetQnnConv2dArgs(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < 3; ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        std::cout << "Debug:" << out.name << std::endl;
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

    if (pattern_name == "beaivi.conv2d") {
      Callables arguments = Conv2d_bias(callee, caller, false);
      return GenerateBody(caller, "beaivi_conv2d_int8_nhwc", arguments);
    } else if (pattern_name == "beaivi.conv2d_depthwise") {
      Callables arguments = Conv2d_bias(callee, caller, true);
      return GenerateBody(caller, "beaivi_conv2d_depthwise_int8_nhwc", arguments);
    } else if (pattern_name == "beaivi.avg_pool2d") {
      Callables arguments = Avg_pool2d(callee, caller, false);
      return GenerateBody(caller, "beaivi_avgpool2d_int8_nhwc", arguments);
    } else if (pattern_name == "beaivi.dense1d") {
      Callables arguments = Dense_1d(callee, caller, false);
      return GenerateBody(caller, "beaivi_dense1d_int8", arguments);
    }

    LOG(FATAL) << "Unknown composite function:" << pattern_name;
  }

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const Callables& args) {
    // Make function call with input buffers when visiting arguments
    // ICHECK_GT(func_args.size(), 0);
    std::ostringstream decl_stream;

    // Wildcard arguments i.e. input, weight, output
    if (args.input.size() > 0) {
      decl_stream << "(" << args.input[0];
      std::cout << "Input arguments handled" << std::endl;
    }

    // Analyze the output buffers
    std::vector<Type> out_types;
    if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = root_call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(root_call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
    }

    // Create a work buffer for paddings

    std::cout << "Output buffers analyzed" << std::endl;

    // Generate buffers to hold results
    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      this->PrintIndents();
      std::string out;
      // Attach to module output buffer
      if (++buf_idx_ == funcs_in_mod_) {
        out = "out0";
      } else {
        out = "io_buf";
      }
      const auto out_size = GetShape1DSize(out_type) * sizeof(int32_t);

      decl_stream << ", " << out;

      if (args.layer == "conv2d" || args.layer == "dense1d") {
        decl_stream << ", " << "padding_buf";
      }

      Output output;
      output.name = out;
      output.size = out_size;
      output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
      output.need_copy = false;
      ret.buffers.push_back("int* " + out + " = (int*)malloc(" + std::to_string(out_size) + ");");
      ret.outputs.push_back(output);
    }

    if (args.weights.size() > 0) {
      decl_stream << ", " << args.weights[0];
    }

    if (args.bias.size() > 0) {
      decl_stream << ", " << args.bias[0];
    }

    // Input subtraction
    //
    if (args.input_sub_attr.size() > 0) {
      std::cout << "Input sub: " << args.input_sub_attr[0] << std::endl;
      decl_stream << ", " << args.input_sub_attr[0];
    }

    // Conv2d Attrs
    for (size_t i = 0; i < args.conv2d_attrs.size(); ++i) {
      decl_stream << ", " << args.conv2d_attrs[i];
    }

    // Avg pool attrs
    for (size_t i = 0; i < args.pool_attrs.size(); ++i) {
      decl_stream << ", " << args.pool_attrs[i];
    }

    // Dense 1d attrs
    for (size_t i = 0; i < args.dense_attrs.size(); ++i) {
      decl_stream << ", " << args.dense_attrs[i];
    }

    // // Requantize attrs
    if (args.scales.size() > 0) {
      decl_stream << ", " << args.scales[0];  // Scaling factor
    }
    if (args.shifts.size() > 0) {
      decl_stream << ", " << args.shifts[0];  // Shift
    }
    if (args.zp_subtract.size() > 0) {
      decl_stream << ", " << args.zp_subtract[0];  // Input Zero point
    }
    decl_stream << ");";
    ret.decl = func_name + decl_stream.str();
    return ret;
  }

  /*!
   * \brief The accumulated constant name to constant mapping. Shared between all generated
   * functions.
   */
  std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant_;
  std::map<std::string, ExtractedConstArray> extracted_constants;

  /*! \brief The id of the external dnnl ext_func. */
  std::string ext_func_id_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The index of global constants. */
  int const_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using DNNL kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration of intermeidate buffers. */
  std::vector<std::string> buf_decl_;
  std::vector<int> work_buffers_;
  /*! \brief The variable name to constant mapping. */
  Array<String>* const_names_;
  int funcs_in_mod_{0};

  friend class BeaiviModuleCodegen;
};

class BeaiviModuleCodegen : public CSourceModuleCodegenBase {
 public:
  // Create a corresponding DNNL function for the given relay Function.
  std::pair<std::string, Array<String>> GenBeaiviFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);
    func_names_.push_back(sid);

    CodegenBeaivi builder(&const_name_to_constant_, &const_names_, sid);
    auto out = builder.VisitExpr(func->body);
    code_stream_ << builder.JIT(out);

    return {sid, const_names_};
  }

  /*! \brief Returns the accumulated constant name to constant mapping. */
  const std::unordered_map<std::string, runtime::NDArray>& const_name_to_constant() const {
    return const_name_to_constant_;
  }

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and dnnl specific ones. To make
   * linking simpiler, the DNNL kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/dnnl folder.
   *
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <stdint.h>\n";
    code_stream_ << "#include <dsp_conv2d.h>\n";
    code_stream_ << "#include <dsp_conv2d_depthwise.h>\n";
    code_stream_ << "#include <dsp_avgpool2d.h>\n";
    code_stream_ << "#include <dsp_dense1d.h>\n";
    code_stream_ << "\n";

    ICHECK(ref->IsInstance<FunctionNode>());
    auto res = GenBeaiviFunc(Downcast<Function>(ref));
    std::string code = code_stream_.str();
    String sym = std::get<0>(res);
    Array<String> variables = std::get<1>(res);

    std::cout << "Sym: " << sym << std::endl;

    int i = 0;
    for (auto x : variables) {
      std::cout << i << " | " << "Var: " << x << std::endl;
      ++i;
    }

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    //// TODO(@manupa-arm): pass the function names to enable system-lib creation
    // return (*pf)(code, "c", Array<String>{sym}, variables);
    //  Use this if things break
    return codegen::CSourceModuleCreate(code, "c", func_names_);
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
  Array<String> func_names_;
  std::unordered_map<std::string, runtime::NDArray> const_name_to_constant_;
  Array<String> const_names_;
};

runtime::Module BeaiviCompiler(const ObjectRef& ref) {
  BeaiviModuleCodegen beaivi;
  return beaivi.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.beaivi").set_body_typed(BeaiviCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
