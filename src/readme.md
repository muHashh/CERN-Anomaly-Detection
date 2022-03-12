# Summer Project Code


## How to run the code

Ensure the required packages are installed:

* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **Pandas** ([instructions](http://pandas.pydata.org/pandas-docs/stable/install.html))
* **NumPy** ([instructions](https://docs.scipy.org/doc/numpy/user/install.html))
* **Scikit-Learn** ([instructions](https://scikit-learn.org/stable/install.html))
* **QKeras** ([instructions](https://github.com/google/qkerasl))

Then process the data

```
python create_datasets.py --qcd="../../data/bkg_3mln.h5" --signals="../../data/sig*" --qcd_out="./dataset" --signals_out="./signals" --no-scale
```

Use the `--scale` if you'd like to scale the data.

The dataset will be stored in `dataset` and and the processed signal data in `signals` as HDF5 files. From there you can start the training.

```
python python train.py --model garnet --signals signals --dataset= dataset --out output/garnet --quant_size 0 --pruning False --latent_dim 8 --device 0
```

This is just an example command, you can learn more about the options by entering

```
python train.py -h
```

The results of training (ROC curves, loss, etc.) can then be viewed in `notebooks/plot_results.ipynb`.
