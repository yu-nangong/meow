"""Quick sanity test: does the MLP actually parse and run on a small subset?"""
import os
os.environ["MEOW_MODEL_TYPE"] = "mlp"
os.environ["MEOW_MLP_EPOCHS"] = "2"
os.environ["MEOW_MLP_HIDDEN1"] = "16"
os.environ["MEOW_MLP_HIDDEN2"] = "8"
os.environ["MEOW_N_CHUNKS"] = "1"

from data_io import load_day, default_h5dir
from feat import MeowFeatureGenerator
from models.tiny_mlp import TinyMLP
import numpy as np

h5dir = default_h5dir()
print(f"Data dir: {h5dir}")

raw = load_day(h5dir, 20230601)
print(f"Loaded {len(raw)} rows")

feat_gen = MeowFeatureGenerator(cacheDir=None)
xdf, ydf = feat_gen.genFeatures(raw)
print(f"Features: {xdf.shape[1]} columns, {len(xdf)} rows")
print(f"Feature names (first 5): {list(xdf.columns[:5])}")

model = TinyMLP(cacheDir=None)
X = xdf.to_numpy(dtype=np.float64)
y = ydf.to_numpy(dtype=np.float64).ravel()
model.fit_from_arrays(X[:10000], y[:10000])
pred = model.predict(xdf.iloc[:100])
print(f"Predictions (first 5): {pred[:5]}")
print(f"Pearson: {float(np.corrcoef(y[:10000], model.predict(xdf.iloc[:10000]))[0, 1]):.6f}")
print("MLP works!")
