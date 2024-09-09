import argparse
import os
import re
import textwrap
from tvm import relay
import tvm
from tvm import runtime
#from tvm.relay.op.contrib.register import get_pattern_table
#from tvm.relay.op.contrib.headsail import legalize_qnn_for_headsail
from tvm.micro import export_model_library_format
import tvm.micro
import logging
import onnx
import numpy as np
from pathlib import Path
#from tiny_perf_benchmark import get_ic_stimulus, get_kws_stimulus, get_vww_stimulus
from backend import HeadsailBackend

#tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)

SHAPES = {
    "mobilenet": {"input" :(1, 3, 224, 224)},
    "conv2dbasic": {"input" :(1, 3, 32, 32)},
    "add": {"Input1": (1), "Input2": (1)},
    "perf_ic": {"input_1_int8": (1, 32, 32, 3)},
    "perf_vww": {"data": (1, 96, 96, 3)},
    "perf_kws": {"data": (1, 49, 10, 1)},
}

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def quantize(mod, params, quantize_arg="int8"):
    print("Quantizing model")
    with relay.quantize.qconfig(calibrate_mode='global_scale', dtype_input=quantize_arg, dtype_weight=quantize_arg, dtype_activation=quantize_arg, global_scale=8.0):
        modd = relay.quantize.quantize(mod, params)
        return modd

def write_c_stimulus(data, len_symbol="stimulus_c_len", payload_symbol="stimulus_c", payload_type="unsigned char"):
    c_file = open("model_c/stimulus" + ".c", "w")
    len_line = "unsigned int {len_symbol} = {data_len};\n".format(len_symbol=len_symbol, data_len=len(data))

    print(data)
    data_str = ""
    for x in data:
        data_str += str(x) + ", "
    data_str = data_str[:-2]

    payload = textwrap.fill(data_str, 80, initial_indent="  ", subsequent_indent="  ")
    payload_line = "{payload_type} {payload_symbol}[] = {{\n{payload}}};\n".format(payload_type=payload_type,
                                                                                   payload_symbol=payload_symbol,
                                                                                   payload=payload)
    c_file.write(len_line + payload_line)
    c_file.close()
    print("Stimulus written!")


def import_onnx_model(path_to_model, shape_dict):
    onnx_model = onnx.load(path_to_model)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    return mod, params

def import_pytorch_model(path_to_model, shape_dict):
    shape_list = []
    for input_key in shape_dict:
        shape_list.append((input_key, shape_dict[input_key]))

    onnx_model = onnx.load(path_to_model)
    mod, params = relay.frontend.from_pytorch(onnx_model, shape_list)
    return mod, params

def import_tflite_model(path_to_model, shape_dict, input_type):
    import tensorflow as tf
    tflite_model_buf = open(path_to_model, "rb").read()

    try:
        import tflite
        tf_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.model
        tf_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    dtype_dict = {}
    for input_key in shape_dict:
        dtype_dict[input_key] = input_type

    print(shape_dict)
    print(dtype_dict)

    mod, params = relay.frontend.from_tflite(tf_model, shape_dict, dtype_dict)
    return mod, params

def import_tf_model(path_to_model, shape_dict, input_type):
    dtype_dict = {}
    for input_key in shape_dict:
        dtype_dict[input_key] = input_type

    mod, params = relay.frontend.from_tensorflow(path_to_model, shape_dict, dtype_dict)
    return mod, params


# def headsail_annotate(mod, params):
#     print("Annotating graph for Headsail DLA")
#     headsail_patterns = get_pattern_table("headsail")

#     #from tvm.relay.quantize.quantize import _bind_params
#     #mod["main"] = _bind_params(mod["main"], params)
#     mod = legalize_qnn_for_headsail(mod)
#     annotation_pass = tvm.transform.Sequential(
#         [
#             #relay.transform.InferType(),
#             #relay.transform.SimplifyInference(),
#             #relay.transform.FoldConstant(),
#             #relay.transform.FoldScaleAxis(),
#             # relay.transform.Legalize('FTVMQnnLegalize'),
#             # relay.transform.Legalize('FTVMQnnCanonicalize'),
#             # relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'OIHW']}),
#             # relay.transform.Legalize(),
#             # relay.transform.AlterOpLayout(),
#             relay.transform.MergeComposite(headsail_patterns),
#             relay.transform.AnnotateTarget(["headsail"]), # Output: Figure 2
#             relay.transform.MergeCompilerRegions(), # Output: Figure 3
#             relay.transform.PartitionGraph(), # Output: Figure 4
#             # GenerateCMSISNNConstants(),
#             # CMSISNNFusePads(),
#             # ScalarToTensorConstants(),
#             # ExtractConstantsFromPartitionedFunction(),
#             relay.transform.InferType(),
#         ])
#     mod = annotation_pass(mod)

#     return mod

def export_annotated_library(mod, params, build_dir):

    mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
    print(tvm.target.Target.list_kinds())
    file_format_str = "{name}_c.{ext}"
    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib" : False})
    # TARGET = tvm.target.Target("llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mabi=lp64 -mattr=+64bit")
    # target_c = tvm.target.Target("llvm -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mabi=lp64 -mattr=+64bit")
    TARGET = tvm.target.Target("c")

    uma_backend = HeadsailBackend()
    uma_backend.register()

    headsail_target = tvm.target.Target(uma_backend.target_name, host=TARGET)
    target_c = tvm.target.Target("c")
    EXECUTOR = tvm.relay.backend.Executor("aot", {"unpacked-api": True, "interface-api": "c", "link-params": False})

    #mod = uma_backend.partition(mod, params)
    mod = uma_backend.partition(mod, params)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = relay.build(mod, target=[target_c, headsail_target], runtime=RUNTIME, params=params, executor=EXECUTOR)

    lib_file_name = os.path.join(build_dir, file_format_str.format(name="model", ext="tar"))
    export_model_library_format(lib, lib_file_name)
    return lib, lib_file_name, mod

def generate_hex_dumps(lib_file_name, build_dir):
    owd = os.getcwd()
    os.chdir(build_dir)
    os.system("tar -xvf {lib}".format(lib=lib_file_name))
    # graph_path_rel = os.path.relpath("{name}_c.{ext}".format(name="graph", ext="json"))
    # params_path_rel = os.path.relpath("{name}_c.{ext}".format(name="params", ext="bin"))
    # os.system("xxd -i {graph} > {graphc} ".format(graph=graph_path_rel, graphc=(graph_path_rel + ".c")))
    # os.system("xxd -i {params} > {paramsc} ".format(params=params_path_rel, paramsc=(params_path_rel + ".c")))
    os.chdir(owd)

def export_stimulus(stimulus, input_type):
    _, stim_ext = os.path.splitext(stimulus)
    if stim_ext == ".npy":
        stimulus_path = Path(__file__).parents[0] / stimulus
        data = np.load(stimulus_path).flatten()
        data = normalize(data)
        write_c_stimulus(data, payload_type=input_type)

def build_model(opts, shape_dict):

    # Make sure that TVM can find headsail codegen if annotation is used
    # if opts.annotate_graph and not tvm.get_global_func("relay.ext.headsail", True):
    #     print("skip because headsail codegen is not available")
    #     return

    # Create build dir for model files
    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)

    model_ext = os.path.splitext(opts.model_path)[1]
    if model_ext == ".onnx":
        mod, params = import_onnx_model(opts.model_path, shape_dict)
    elif model_ext == ".pth":
        mod, params = import_pytorch_model(opts.model_path, shape_dict)
    elif model_ext == ".tflite":
        mod, params = import_tflite_model(opts.model_path, shape_dict, opts.input_type)
    elif model_ext == ".tf":
        mod, params = import_tf_model(opts.model_path, shape_dict, opts.input_type)
    else:
        print("Error! Unsupported model", opts.model_path)
        return

    #TVM quantization (only for tflite?)
    # if opts.annotate_graph:
    #     mod = quantize(mod, params)

    # Annotate model with headsail tags
    # if opts.annotate_graph:
    #     mod = headsail_annotate(mod, params)
    # print(mod)


    # Export library
    lib, lib_file_name, mod = export_annotated_library(mod, params, build_dir)
    file_format_str = "{name}_c.{ext}"
    # Write mod log to output
    with open(
            os.path.join(build_dir, "mod_output.txt"), "w"
        ) as mod_log:
            mod_log.write(str(mod))


    # Export graph
    # with open(
    #     os.path.join(build_dir, file_format_str.format(name="graph", ext="json")), "w"
    # ) as f_graph_json:
    #     f_graph_json.write(lib.get_graph_json())

    # print("Params", params)
    # # Export weights
    # with open(
    #     os.path.join(build_dir, file_format_str.format(name="params", ext="bin")), "wb"
    # ) as f_params:
    #     f_params.write(runtime.save_param_dict(lib.get_params()))

    # Generate stimulus
    if opts.stimulus != None:
        opts.stimulus = opts.stimulus.strip()
        export_stimulus(opts.stimulus, opts.input_type)
    os.chdir(build_dir)

    # Convert graph and weights to hexdumps
    generate_hex_dumps(lib_file_name, build_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="./build")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--model_path", required=True)
    parser.add_argument("-s", "--stimulus", default=None)
    parser.add_argument("-t", "--input_type", default="unsigned char")
    parser.add_argument("--annotate-graph", action="store_true")

    opts = parser.parse_args()

    if opts.model == "add":
        build_model(opts, shape_dict=SHAPES["add"])
    elif opts.model == "mobilenet":
        build_model(opts, shape_dict=SHAPES["mobilenet"])
    elif opts.model == "conv2dbasic":
        build_model(opts, shape_dict=SHAPES["conv2dbasic"])
    elif opts.model == "perf_image_classification":
        build_model(opts, shape_dict=SHAPES["perf_ic"])
        os.chdir(os.path.abspath("../"))
        write_c_stimulus(get_ic_stimulus(), "int8_t")
    elif opts.model == "perf_visual_wakeup_word":
        build_model(opts, shape_dict=SHAPES["perf_vww"])
        os.chdir(os.path.abspath("../"))
        write_c_stimulus(get_vww_stimulus(), "int8_t")
    elif opts.model == "perf_keyword_spotting":
        build_model(opts, shape_dict=SHAPES["perf_kws"])
        os.chdir(os.path.abspath("../"))
        write_c_stimulus(get_kws_stimulus(), "int8_t")
    else:
        print("No such model", opts.model, "Availeable models: add, mobilenet")
