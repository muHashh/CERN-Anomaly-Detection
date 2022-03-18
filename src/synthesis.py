# hls4ml version used: pip install git+https://github.com/muHashh/hls4ml.git@jet_tag_paper

import numpy as np
import hls4ml
# import plotting
import tensorflow as tf
# from sklearn.metrics import accuracy_score
from models import garnet_ae
import h5py
from external_models.garnet_old import GarNet
from tensorflow.keras.models import load_model
import utils.plotting as plotting
# from qkeras import QActivation, QDense, QConv1D, QConv2D, quantized_bits
import os
import glob
from pathlib import Path


os.environ["PATH"] = "/mnt/data/tools/Xilinx/Vivado/2020.1/bin:" + os.environ["PATH"]

model = load_model("../output/test16/model.h5", custom_objects={"GarNet": GarNet})


config = hls4ml.utils.config_from_keras_model(model, granularity="model")
# np.save("config.npy", config)

# config = np.load("config.npy", allow_pickle='TRUE').item()

print("-----------------------------------")
print("Configuration")
print("-----------------------------------")

out = "hls_test16"
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


print("\n\n\nRunning Predictions...\n")

dataset = "../datasets/dataset1"
signals = "../signals/signals1"

dataset = h5py.File(dataset+"/dataset.h5", "r")
x_test = dataset["x_test"][()]
X_test = (x_test, np.ones((x_test.shape[0], 1))*x_test.shape[1])

y_hls = hls_model.predict(X_test)
# y_hls = np.expand_dims(np.reshape(y_hls, (np.shape(y_hls)[0],) + (16, 3)), axis=3)

prediction = h5py.File(out + "/predictions.h5", "w")
prediction.create_dataset('QCD', data=x_test)
prediction.create_dataset('predicted_QCD', data=y_hls)

for signal in glob.glob(signals+"/*"):
    signal_jets = h5py.File(signal, 'r')["jetConstituentsList"][()]
    signal_jets = (signal_jets, np.ones((signal_jets.shape[0], 1))*signal_jets.shape[1])

    y_anomaly = hls_model.predict(signal_jets)
    # y_anomaly = np.expand_dims(np.reshape(y_anomaly, (np.shape(y_anomaly)[0],) + (16, 3)), axis=3)
    prediction.create_dataset('predicted_'+Path(signal).stem, data=y_anomaly)

prediction.close()

print("\n\n\nDone\n")
# print(hls4ml.templates.get_supported_boards_dict().keys())
# plotting.print_dict(hls4ml.templates.get_backend('VivadoAccelerator').create_initial_config())


hls_model.build(reset=True, csim=True, cosim=True, synth=True, vsynth=True)
# hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)

hls4ml.report.read_vivado_report(out)

