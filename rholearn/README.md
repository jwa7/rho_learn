This subpackage is concerned with the training of ML models for the prediction
of the electron density.

Training is performed with custom `torch` modules, interfaced with equistore.
This subpackage therefore works with inputs and outputs in the
`equistore.TensorMap` format.

Reference / QM data generated with electronic structure codes can be parsed into
equistore format using the `rhoparse` subpackage.