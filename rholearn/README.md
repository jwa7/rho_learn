This subpackage is concerned with the training of ML models for the prediction
of the electron density.

Training is performed with custom `torch` modules, interfaced with metatensor.
This subpackage therefore works with inputs and outputs in the
`metatensor.TensorMap` format.

Reference / QM data generated with electronic structure codes can be parsed into
metatensor format using the `rhoparse` subpackage.