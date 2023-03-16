import os
import pprint
import ase.io
import numpy as np

import equistore   # storage format for atomistic ML
from equistore import Labels

from equisolve.utils import split_data

from rholearn import io, features, utils
from settings import RASCAL_HYPERS, DATA_SETTINGS, ML_SETTINGS


# Read the water molecules from file
n_structures = 1000
frames = ase.io.read(DATA_SETTINGS["xyz"], index=f":{n_structures}"
)

# Compute lambda-SOAP: uses rascaline to compute a SphericalExpansion
# Runtime approx 25 seconds
input = features.lambda_soap_vector(
    frames, RASCAL_HYPERS, even_parity_only=True
)

# Load the electron density data
output = equistore.load(os.path.join(DATA_SETTINGS["data_dir"], "e_densities.npz"))

# Drop the block for l=5, Hydrogen as this isn't included in the output electron density
input = equistore.drop_blocks(input, keys=np.setdiff1d(input.keys, output.keys))

# Check that the metadata of input and output match along the samples and components axes
assert equistore.equal_metadata(input, output, check=["samples", "components"])

# Save lambda-SOAP descriptor to file
equistore.save(os.path.join(DATA_SETTINGS["data_dir"], "lambda_soap.npz"), input)

# Write settings to file
with open(os.path.join(DATA_SETTINGS["data_dir"], "rascal_hypers.txt"), "w+") as f:
    f.write(f"Rascal hypers:\n{pprint.pformat(RASCAL_HYPERS)}\n")
with open(os.path.join(DATA_SETTINGS["data_dir"], "data_settings.txt"), "w+") as f:
    f.write(f"Data settings:\n{pprint.pformat(DATA_SETTINGS)}\n")

# Split the data into training, validation, and test sets
[[in_train, in_test, in_val], [out_train, out_test, out_val]], grouped_labels = split_data(
    [input, output],
    axis=DATA_SETTINGS["axis"],
    names=DATA_SETTINGS["names"],
    n_groups=DATA_SETTINGS["n_groups"],
    group_sizes=DATA_SETTINGS["group_sizes"],
    seed=DATA_SETTINGS["seed"],
)
tm_files = {
    "in_train.npz": in_train,
    "in_test.npz": in_test,
    "out_train.npz": out_train,
    "out_test.npz": out_test,
    "in_val.npz": in_val,
    "out_val.npz": out_val,
}
# Save the TensorMaps to file
for name, tm in tm_files.items():
    equistore.save(os.path.join(DATA_SETTINGS["data_dir"], name), tm)