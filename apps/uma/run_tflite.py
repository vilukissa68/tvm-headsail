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

import tensorflow as tf
import os

from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from backend import VanillaAcceleratorBackend
from tvm.relay import transform, testing
from collections import OrderedDict
import numpy as np

def import_tf_model(path_to_model, shape_dict, input_type):
    dtype_dict = {}
    for input_key in shape_dict:
        dtype_dict[input_key] = input_type

    mod, params = relay.frontend.from_tensorflow(path_to_model, shape_dict, dtype_dict)
    return mod, params

def main():
    mod, params = import_tf_model(opts.model_path, shape_dict)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()

    target = tvm.target.Target("headsail", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    mod = uma_backend.partition(mod)

    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path


if __name__ == "__main__":
    main()
