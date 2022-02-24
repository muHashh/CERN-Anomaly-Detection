# hls4ml version used: pip install git+https://github.com/muHashh/hls4ml.git@jet_tag_paper

import numpy as np
import hls4ml
# import plotting
import tensorflow as tf
# from sklearn.metrics import accuracy_score
from models import garnet_ae
from external_models.garnet_old import GarNet
from tensorflow.keras.models import load_model
import utils.plotting as plotting
# from qkeras import QActivation, QDense, QConv1D, QConv2D, quantized_bits
import os


os.environ["PATH"] = "/mnt/data/tools/Xilinx/Vivado/2020.1/bin:" + os.environ["PATH"]

model = load_model("./output/garnet_2_layers-2/model.h5", custom_objects={"GarNet": GarNet})


config = hls4ml.utils.config_from_keras_model(model, granularity="model")
# np.save("config.npy", config)

# config = np.load("config.npy", allow_pickle='TRUE').item()

print("-----------------------------------")
print("Configuration")
print("-----------------------------------")

out = "hls_out2"
hls_model = hls4ml.converters.convert_from_keras_model(model, 
                                                       hls_config=config,
                                                       output_dir=out,
                                                       io_type="io_parallel",
                                                       part="xcvu9p-flgb2104-2-e",
                                                       backend="VivadoAccelerator",)

# f = open("convert.obj", "wb")
# pickle.dump(hls_model, f)

# cfg = hls4ml.converters.create_config('xcvu9p-flgb2104-2l-e')

# cfg['IOType']     = 'io_parallel'
# cfg['HLSConfig']  = config
# cfg['KerasModel'] = model
# cfg['OutputDir']  = '{}/{}'.format("./hls4ml", "garnet")

# hls_model = hls4ml.converters.keras_to_hls(cfg)
# hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
hls_model.compile()

# print(hls4ml.templates.get_supported_boards_dict().keys())
# plotting.print_dict(hls4ml.templates.get_backend('VivadoAccelerator').create_initial_config())


hls_model.build(csim=False, synth=True, vsynth=True)
# hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)

hls4ml.report.read_vivado_report(out)

