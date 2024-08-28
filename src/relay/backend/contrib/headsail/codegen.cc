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

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <numeric>
#include <cstring>

#include "../../utils.h"
//#include "codegen_headsail.h"
#include "../codegen_c/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"
#include "../../../transforms/compiler_function_utils.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
	const auto shape = GetShape(type);
	return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}


std::vector<std::string> Conv2d_bias(const CallNode* call) {

	std::vector<std::string> args;

	const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
	ICHECK(conv2d_attr);

	auto ishape = GetShape(call->args[0]->checked_type());  // Input shape
	auto wshape = GetShape(call->args[1]->checked_type());  // Kernel shape

	std::cout << std::endl;
	// args.push_back(std::to_string(ishape[0])); // Input batch
	// std::cout << "Input batch: " << std::to_string(ishape[0]) << std::endl;

	args.push_back(std::to_string(ishape[1]));  // Input height
	std::cout << "Input height: " << std::to_string(ishape[1]) << std::endl;

	args.push_back(std::to_string(ishape[2]));  // Input width
	std::cout << "Input width: " << std::to_string(ishape[2]) << std::endl;

	args.push_back(std::to_string(ishape[3]));  // Input channels
	std::cout << "Input channel: " << std::to_string(ishape[3]) << std::endl;

	// Input layout
	char data_layout[6];
	std::strcpy(data_layout, "\"");
	std::strcat(data_layout, &conv2d_attr->data_layout.c_str()[1]);
	std::strcat(data_layout, "\"");

	std::cout << "Data layout: " << data_layout << std::endl;
	args.push_back(data_layout);

	args.push_back(std::to_string(wshape[0]));  // Kernels amount
	std::cout << "Kernels amount: " << std::to_string(wshape[0]) << std::endl;

	args.push_back(std::to_string(wshape[1]));  // Kernels height
	std::cout << "Kernels height: " << std::to_string(wshape[1]) << std::endl;

	args.push_back(std::to_string(wshape[2]));  // Kernels width
	std::cout << "Kernels width: " << std::to_string(wshape[2]) << std::endl;

	args.push_back(std::to_string(wshape[3]));  // Kernels channels
	std::cout << "Kernels channels: " << std::to_string(wshape[3]) << std::endl;

	// Kernel layout
	char kernel_layout[7];
	std::strcpy(kernel_layout, "\"");
	std::strcat(kernel_layout, conv2d_attr->kernel_layout.c_str());
	std::strcat(kernel_layout, "\"");

	// Convert TVM layout string to Headsail layout string
	for (int i = 0; i < 7; ++i) {
		if (kernel_layout[i] == 'I') {
			kernel_layout[i] = 'K';
		} else if (kernel_layout[i] == 'O') {
			kernel_layout[i] = 'C';
		}
	}
	std::cout << "Kernel layout: " << kernel_layout << std::endl;
	args.push_back(kernel_layout);

	std::cout << "Bias size: " << std::to_string(conv2d_attr->groups * wshape[3]) << std::endl;
	args.push_back(std::to_string(conv2d_attr->groups * wshape[3]));

	// Padding
	args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));  // Pad top
	args.push_back(
		std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));  // Pad left
	args.push_back(
		std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));  // Pad right
	args.push_back(
		std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));  // Pad bottom
	args.push_back(std::to_string(0));                                     // Pad value

	args.push_back(
		std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));  // Stride x
	args.push_back(
		std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));  // Stride y
	args.push_back(std::to_string(0));                                     // Mac clip
	args.push_back(std::to_string(0));                                     // PP clip

	return args;
}

class HeadsailCodegen : public backend::MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
	public:
		HeadsailCodegen(std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant,
						Array<String>* const_names, std::string ext_func_id)
      : const_name_to_constant_(const_name_to_constant),
        const_names_(const_names),
        ext_func_id_(std::move(ext_func_id)) {}

	std::string JIT(const std::vector<Output>& out) override {
		// Write function macros
		for (auto decl : func_decl_) {
			code_stream_ << decl << "\n";
		}
		return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
	}


	private:
		std::vector<Output> VisitExprDefault_(const Object* op) override {
			LOG(FATAL) << "C codegen doesn't support: " << op->GetTypeKey();
		}

		std::vector<Output> VisitExpr_(const VarNode* node) override {
			ext_func_args_.push_back(GetRef<Var>(node));
			Output output;
			output.name = node->name_hint();
			return {output};
		}

		std::vector<Output> VisitExpr_(const TupleNode* node) override {
			std::vector<Output> outs;
			for (auto field : node->fields) {
				auto res = VisitExpr(field);
				ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
				outs.push_back(res[0]);
			}
			return outs;
		}

		std::vector<Output> VisitExpr_(const TupleGetItemNode* op) override {
			auto res = VisitExpr(op->tuple);
			ICHECK_GT(res.size(), static_cast<size_t>(op->index));
			// Only keep the item we want for the child node.
			// FIXME(@comaniac): The other items should still be requried for the primary outputs.
			return {res[op->index]};
		}

		std::vector<Output> VisitExpr_(const ConstantNode* cn) override {
			// Remember we'll need some extra headers to support the runtime constants array.

			std::ostringstream decl_stream;
			std::ostringstream buf_stream;

			Output output;
			// Get const: static_cast<float*>(gcc_0_consts[0]->data)
			size_t const_id = const_name_to_constant_->size();
			output.name = CreateDataReference(ext_func_id_, const_id);
			const auto* type_node = cn->checked_type().as<TensorTypeNode>();
			ICHECK(type_node);
			const auto& dtype = GetDtypeString(type_node);

			// Generate the global variable for needed ndarrays
			if (const_array_name_.empty()) {
				const_array_name_ = CreateNDArrayPool(ext_func_id_);
				std::string checker = CreateInitChecker(ext_func_id_);
				ext_func_body_.insert(ext_func_body_.begin(), checker);
			}

			ICHECK(dtype == "float" || dtype == "int") << "Only float and int are supported for now.";
			output.dtype = dtype;

			std::string const_var_name = CreateConstVar(ext_func_id_, const_id);
			const_name_to_constant_->emplace(const_var_name, cn->data);
			const_names_->push_back(const_var_name);

			return {output};
		}

		std::vector<Output> VisitExpr_(const CallNode* call) override {
			std::ostringstream macro_stream;
			std::ostringstream decl_stream;
			std::ostringstream buf_stream;

			std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

			// Make function declaration
			macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

			if (backend::IsOp(call, "headsail.tflite_conv2d_bias_relu")) {
				auto args = Conv2d_bias(call);
				for (auto arg: args) {
					macro_stream << arg << " ";
				}
			} else if (backend::IsOp(call, "subtract")) {
				macro_stream << "-";
			} else if (backend::IsOp(call, "multiply")) {
				macro_stream << "*";
			} else {
			LOG(FATAL) << "Unrecognized op";
			}

			auto in_shape = backend::GetShape(call->args[0]->checked_type());
			for (size_t i = 0; i < in_shape.size(); ++i) {
			macro_stream << ", " << in_shape[i];
			}

			const auto* type_node = call->checked_type().as<TensorTypeNode>();
			ICHECK(type_node);
			const auto& dtype = GetDtypeString(type_node);
			macro_stream << ", " << dtype;

			macro_stream << ");";
			func_decl_.push_back(macro_stream.str());

			// Make function call when visiting arguments
			bool first = true;
			decl_stream << func_name << "(";
			for (size_t i = 0; i < call->args.size(); ++i) {
			auto res = VisitExpr(call->args[i]);
			for (auto out : res) {
				if (!first) {
					decl_stream << ", ";
				}
					first = false;
					decl_stream << out.name;
			    }
			}

			std::string out = "buf_" + std::to_string(buf_idx_++);
			auto out_shape = backend::GetShape(call->checked_type());
			int out_size = 1;
			for (size_t i = 0; i < out_shape.size(); ++i) {
			out_size *= out_shape[i];
			}
			buf_stream << dtype << "* " << out << " = (" << dtype << "*)malloc(4 * " << out_size << ");";
			buf_decl_.push_back(buf_stream.str());

			decl_stream << ", " << out << ");";
			ext_func_body_.push_back(decl_stream.str());

			// Update output buffer
			// Note C codegen only handles TensorType. Therefore, we don't flatten
			// tuples and only return a single vaule.
			Output output;
			output.name = out;
			output.dtype = dtype;
			output.need_copy = true;
			output.size = out_size;
			return {output};
		}


		/*!
		* \brief The accumulated constant name to constant mapping. Shared between all generated
		* functions.
		*/
		std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant_;
		/*! \brief The accumulated constant names, in the order they were generated. */
		Array<String>* const_names_;
		/*! \brief Name of the global function currently being compiled. */
		std::string ext_func_id_;
		/*! \brief The index of the next available wrapped C function. */
		int func_idx = 0;
		/*! \brief The index of the next available allocated buffers. */
		int buf_idx_ = 0;
		/*! \brief The arguments of a C compiler compatible function. */
		Array<Var> ext_func_args_;
		/*! \brief The statements of a C compiler compatible function. */
		std::vector<std::string> ext_func_body_;
		/*! \brief The array declared to store the constant values. */
		std::string const_array_name_;
		/*! \brief The declaration statements of a C compiler compatible function. */
		std::vector<std::string> func_decl_;
		/*! \brief The declaration statements of buffers. */
		std::vector<std::string> buf_decl_;
};

class HeadsailCodegenModule {
    public:
		HeadsailCodegenModule(Target target, IRModule mod) : target_(std::move(target)), mod_(std::move(mod)) {}

		runtime::Module CreateCSourceModule() {
			for (const auto& kv : mod_->functions) {
				if (const auto* function_node = GetHeadsailCompilerFunctionNode(kv.second)) {
						GenCFunc(GetRef<Function>(function_node));
				}
			}
			return Finalize();
		}

    /*! \brief Returns the accumulated constant name to constant mapping. */
	const std::unordered_map<std::string, runtime::NDArray>& const_name_to_constant() const {
		return const_name_to_constant_;
	}

	private:
		/*! \brief Emits the standard C/C++ header into \p os. */
		void EmitPreamble(std::ostringstream& os) {
			// Custom header, if any.
			std::cout << "EmitPreamble!!!!!!!!" << std::endl;
			Optional<String> header = target_->GetAttr<String>("header");
			if (header.defined() && !header.value().empty()) {
				os << header.value().c_str() << "\n";
			}

			// Standard includes.
			os << "#include <stdio.h>\n";
			os << "#include <stdlib.h>\n";
			os << "#include <string.h>\n";
			os << "#include <tvm/runtime/c_runtime_api.h>\n";
			os << "#include <tvm/runtime/c_backend_api.h>\n";
		}

		void GenCFunc(const Function& function) {
			ICHECK(function.defined()) << "Input error: expect a Relay function.";
			std::string ext_func_id = backend::GetExtSymbol(function);
			HeadsailCodegen builder(&const_name_to_constant_, &const_names_, ext_func_id);
			std::vector<Output> out = builder.VisitExpr(function->body);
			code_stream_ << builder.JIT(out);
			func_names_.push_back(ext_func_id);
		}

	/*! \brief Returns function if it is tagged with "Compiler=ccompiler". */
		static const FunctionNode* GetHeadsailCompilerFunctionNode(const Expr& expr) {
			if (const auto* function_node = expr.as<FunctionNode>()) {
				Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
				if (opt_compiler.defined() && opt_compiler.value() == "headsailcompiler") {
					return function_node;
				}
			}
			return nullptr;
		}

		runtime::Module Finalize() {
			std::cout << "Finalize!!!!!!!!" << std::endl;
			std::ostringstream os;
			EmitPreamble(os);
			os << code_stream_.str();
			std::string code = os.str();

			VLOG(1) << "HeadsailCodegenModule generated:" << std::endl << code;

			// Create a CSource module
			const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
			ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
			return (*pf)(code, "c", func_names_, const_names_);
		}

		/*! \brief "ccompiler" Target with compilation options to use. */
		Target target_;
		/*! \brief Module we are compiling. */
		IRModule mod_;
		/*! \brief The accumulated constant name to constant mapping. */
		std::unordered_map<std::string, runtime::NDArray> const_name_to_constant_;
		/*! \brief The accumulated constant names, in the order they were generated. */
		Array<String> const_names_;
		/*! \brief The accumulated function names. */
		Array<String> func_names_;
		/*!
		* \brief The accumulated code stream containing all function definitions.
		* (Does not include the preamble.)
		*/
		std::ostringstream code_stream_;
};

/*! \brief Return the "ccompiler" Target instance to use to guide compilation. */
Target GetHeadsailCompilerTarget() {
	Target target = Target::Current(/*allow_not_defined=*/true);
	if (!target.defined() || target->kind->name != "headsail") {
		// Use the default compilation options if no specific "ccompiler" target was given
		// in the overall targets list. In that case target_hooks.cc will invoke the custom pass
		// without pushing any target instance onto the implicit target stack.
		target = Target("headsail");
	}
	return target;
}


/*! \brief The actual translation pass. */
tvm::transform::Pass HeadsailCompilerImpl() {
	std::cout << "Here!!!!!!!!" << std::endl;
	auto pass_func = [=](IRModule mod, const tvm::transform::PassContext& pass_ctx) {
		VLOG(1) << "HeadsailCompilerImpl input:" << std::endl << PrettyPrint(mod);
		std::cout << "Here!!!!!!!!" << std::endl;
		Target target = GetHeadsailCompilerTarget();

		// Emit the C/C++ code and package it as a CSourceModule.
		HeadsailCodegenModule codegen(target, mod);
		runtime::Module runtime_mod = codegen.CreateCSourceModule();

		// Capture the new runtime module.
		Array<runtime::Module> external_mods =
			mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});
		external_mods.push_back(runtime_mod);

		// Capture the new constants.
		Map<String, runtime::NDArray> const_name_to_constant =
			mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant).value_or({});
		for (const auto& kv : codegen.const_name_to_constant()) {
			ICHECK_EQ(const_name_to_constant.count(kv.first), 0);
			const_name_to_constant.Set(kv.first, kv.second);
		}

    return WithAttrs(mod, {{tvm::attr::kExternalMods, external_mods},
                           {tvm::attr::kConstNameToConstant, const_name_to_constant}});
	};
	std::cout << "Here!!!!!!!!" << std::endl;
	return tvm::transform::CreateModulePass(pass_func, 0, "HeadsailCompilerImpl", {});
}

tvm::transform::Pass HeadsailCompilerPass() {
	return transform::Sequential(
		{transform::OutlineCompilerFunctionsWithExistingGlobalSymbols("headsailc"), HeadsailCompilerImpl(),
		 transform::MarkCompilerFunctionsAsExtern("headsail")});
}


// TVM_REGISTER_TARGET_KIND("headsail", kDLCPU)
//     .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true))
//     .set_attr<relay::transform::FTVMRelayToTIR>(tvm::attr::kRelayToTIR, HeadsailCompilerPass())
//     // Value is prepended to every output CModule.
//     .add_attr_option<String>("header", String(""));

TVM_REGISTER_GLOBAL("relay.ext.headsail").set_body_typed(HeadsailCompilerPass);


}  // namespace contrib
}  // namespace relay
}  // namespace tvm
