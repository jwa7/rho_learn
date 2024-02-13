"""
Module containing the RhoModel class that makes predictions on TensorMaps.
"""
import os
from functools import partial
from typing import Union, Optional, List, Tuple, Callable
import warnings

import ase
import numpy as np
import torch

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from rholearn import predictor


VALID_MODEL_TYPES = ["linear", "nonlinear"]


# ===== RhoModel class for making predictions on TensorMaps


class RhoModel(torch.nn.Module):
    """
    A model that makes equivariant predictions on the TensorMap level.

    The model can be either linear or nonlinear. In the linear case, the
    prediction is made by a single linear layer. In the nonlinear case, the
    prediction is made by passing the equivariant block through a linear layer,
    then element-wise multiplying the output of this with the output of a neural
    network that the corresponding invariant block for that chemical species is
    passed through. The output of this is then passed through a linear output
    layer to make the prediction.

    The model is initialized with a number of arguments. First, the model type
    must be specified as either "linear" or "nonlinear". If "linear", the
    ``hidden_layer_widths``, ``activation_fn``, and ``bias_nn`` arguments are
    ignored. If "nonlinear", these must be specified.

    The ``input`` and ``output`` TensorMaps must be passed, which define the
    metadata of the model. These can be example training data. The samples are
    ignored: only the keys, the components, and the properties are stored, as
    these define the metadata of the model.

    The ``bias_invariants`` argument controls whether or not to use a learnable
    bias in the models for invariant blocks. This applies equally to both linear
    and nonlinear models. If true, a bias is used. If false, no bias is used.

    Altenatively (or additionally), if passed, `invariant_baseline` can be
    passed as a non-learnable bias. This must consist of blocks indexed by keys
    for each invariant block the model is trained on. This is added feature-wise
    to the prediction output by invariant block models, essentially acting like
    a non-learnable bias. This allows invariant block models to essentially
    learn on a baselined quantity, which can be useful for improving the
    accuracy in the case of globally-evaluated loss functions where the
    magnitude of the target property is much larger for invariants than
    covariants.

    In both cases ``bias_invariants`` and ``invariant_baseline``, the bias can
    only be applied to transformations of invariants so equivariance is not
    broken.

    ``descriptor_kwargs`` can be passed as a dict of the settings required to
    build a descriptor from ASE frames, suitable for input to the model. These
    should be the same settings used to generate the data the model was trained
    on. These settings are passed to the function :py:func:`descriptor_builder`
    in module :py:mod:`predictor`, which contains the recipe for building the
    descriptor.

    Similarly, ``target_kwargs`` can be passed as a dict of the settings
    required to build a target from ASE frames and predictions outputted by
    RhoModel.forward(). These are used by the custom function
    :py:func:`target_builder` in module :py:mod:`predictor`, which contains the
    recipe for building the target. For instance, in an indirect learning
    scheme, this function may call a Quantum Chemistry code to calculate a
    derived property.

    If ``global_species`` is passed, these are set and may for instance be used
    to define global correlations in the descriptor builder of the predictor. If
    not passed, the gloabl species are inferred from the keys of the ``input``
    TensorMap.
    """

    # Initialize model
    def __init__(
        self,
        model_type: str,
        input: TensorMap,
        output: TensorMap,
        bias_invariants: bool = False,
        normalize_invariants: bool = False,
        invariant_baseline: Optional[TensorMap] = None,
        hidden_layer_widths: Optional[Union[List[List[int]], List[int]]] = None,
        activation_fn: Optional[torch.nn.Module] = None,
        bias_nn: bool = False,
        descriptor_kwargs: Optional[dict] = None,
        target_kwargs: Optional[dict] = None,
        global_species: Optional[List[int]] = None,
        **torch_settings,
    ):
        super(RhoModel, self).__init__()
        # Set the torch settings
        self._torch_settings = torch_settings
        if self._torch_settings.get("dtype"):
            torch.set_default_dtype(self._torch_settings.get("dtype"))

        # Set the base attributes
        self._set_model_type(model_type)
        self._set_metadata(input, output)
        self._set_biases(bias_invariants)
        self._set_normalize_invariants(normalize_invariants)

        # Set attributes specific to a nonlinear model
        if self._model_type == "nonlinear":
            self._set_hidden_layer_widths(hidden_layer_widths)
            self._set_activation_fn(activation_fn)
            self._set_bias_nn(bias_nn)

        # Passing `invariant_baseline` as a TensorMap will add this back to the
        # predictions made on invariant blocks.
        self._set_invariant_baseline(invariant_baseline)

        # Set the global species if passed
        self._set_global_species(global_species)

        # Set the settings required to build a descriptor from ASE frames and
        # transform the raw prediction of the model
        self._set_descriptor_kwargs(descriptor_kwargs)
        self._set_target_kwargs(target_kwargs)

        # Build the models
        self._set_models()

    @property
    def model_type(self) -> str:
        return self._model_type

    def _set_model_type(self, model_type: str) -> None:
        """
        Sets the "_model_type" attr to either "linear" or "nonlinear"
        """
        assert model_type in VALID_MODEL_TYPES
        self._model_type = model_type

    @property
    def in_metadata(self) -> TensorMap:
        return self._in_metadata

    @property
    def out_metadata(self) -> TensorMap:
        return self._out_metadata

    def _set_metadata(self, input: TensorMap, output: TensorMap) -> None:
        """
        Sets the attributes "_in_metadata" and "_out_metadata" as minimal
        TensorMaps storing the relevant metadata data of input and output.
        These are only defined for the intersection between the input and output
        keys.
        """
        keys = input.keys.intersection(output.keys)
        in_blocks, out_blocks = [], []
        for key in keys:
            assert metatensor.equal_metadata_block(
                input[key], output[key], check=["components"]
            )
            in_block = TensorBlock(
                values=torch.zeros(
                    (
                        1,
                        *[len(c) for c in input[key].components],
                        len(input[key].properties),
                    ),
                    dtype=self._torch_settings.get("dtype"),
                    device=self._torch_settings.get("device"),
                ),
                samples=Labels.single(),
                components=input[key].components,
                properties=input[key].properties,
            )
            out_block = TensorBlock(
                values=torch.zeros(
                    (
                        1,
                        *[len(c) for c in output[key].components],
                        len(output[key].properties),
                    ),
                    dtype=self._torch_settings.get("dtype"),
                    device=self._torch_settings.get("device"),
                ),
                samples=Labels.single(),
                components=output[key].components,
                properties=output[key].properties,
            )
            in_blocks.append(in_block)
            out_blocks.append(out_block)
        self._in_metadata = TensorMap(keys, in_blocks)
        self._out_metadata = TensorMap(keys, out_blocks)

    @property
    def biases(self) -> torch.tensor:
        return self._biases

    def _set_biases(self, bias_invariants: bool) -> None:
        """
        Sets the "_biases" attribute of the model to True for invariant blocks if
        `bias_invariants` is true and false otherwise, and false for all
        covariant (l > 0) blocks.

        This is returned as a list, where each element corresponds to the key
        index stored in self._in_metadata.keys
        """
        if bias_invariants:
            biases = [
                key["spherical_harmonics_l"] == 0 for key in self._in_metadata.keys
            ]
        else:
            biases = [False for key in self._in_metadata.keys]
        self._biases = biases

    @property
    def normalize_invariants(self) -> torch.tensor:
        return self._normalize_invariants

    def _set_normalize_invariants(self, normalize_invariants: bool) -> None:
        """
        Sets the "_normalize_invariants" attribute of the model.

        This is returned as a list, where each element corresponds to the key
        index stored in self._in_metadata.keys
        """
        if normalize_invariants:
            normalize_invariants = [
                key["spherical_harmonics_l"] == 0 for key in self._in_metadata.keys
            ]
        else:
            normalize_invariants = [False for key in self._in_metadata.keys]
        self._normalize_invariants = normalize_invariants

    @property
    def hidden_layer_widths(self) -> List:
        return self._hidden_layer_widths

    def _set_hidden_layer_widths(
        self, hidden_layer_widths: Union[List[List[int]], List[int]]
    ):
        """
        Sets the hidden layer widths for each block, stored in the
        "_hidden_layer_widths" attribute.
        """
        if hidden_layer_widths is None:
            raise ValueError(
                "if ``model_type`` is nonlinear, ``hidden_layer_widths`` must be passed"
            )
        # If passed as a single list, set this as the list for every block
        if isinstance(hidden_layer_widths, List) and isinstance(
            hidden_layer_widths[0], int
        ):
            hidden_layer_widths = [
                hidden_layer_widths for key in self._in_metadata.keys
            ]
        # Check it is now a list of list of int
        assert isinstance(hidden_layer_widths, List) and isinstance(
            hidden_layer_widths[0], List
        )
        self._hidden_layer_widths = hidden_layer_widths

    @property
    def activation_fn(self) -> str:
        return self._activation_fn

    def _set_activation_fn(self, activation_fn):
        """
        Sets the activation function used in the nonlinear model.
        """
        if activation_fn is None:
            raise ValueError(
                "if ``model_type`` is nonlinear, ``activation_fn`` must be passed"
            )
        self._activation_fn = activation_fn

    @property
    def bias_nn(self) -> str:
        """
        Returns whether a bias is used in the nonlinear mutliplier for each
        equivariant block.
        """
        return self._bias_nn

    def _set_bias_nn(self, bias_nn: bool):
        """
        Sets whether a bias is used in the nonlinear mutliplier for each
        equivariant block.
        """
        self._bias_nn = bias_nn

    @property
    def models(self) -> torch.nn.ModuleList:
        """
        Returns all the block models
        """
        return self._models

    def _set_models(self) -> None:
        """
        Builds a model for each block and stores them as a torch ModuleList in
        the "_models" attribute.
        """
        tmp_models = []
        for key_i, key in enumerate(self._in_metadata.keys):
            if self._model_type == "linear":
                if self._biases[key_i]:
                    assert key["spherical_harmonics_l"] == 0
                block_model = _LinearModel(
                    in_features=len(self._in_metadata[key].properties),
                    out_features=len(self._out_metadata[key].properties),
                    bias=self._biases[key_i],
                )

            else:
                assert self._model_type == "nonlinear"
                if self._biases[key_i]:
                    assert key["spherical_harmonics_l"] == 0
                in_invariant_block = self._in_metadata.block(
                    spherical_harmonics_l=0, species_center=key["species_center"]
                )
                block_model = _NonLinearModel(
                    in_features=len(self._in_metadata[key].properties),
                    out_features=len(self._out_metadata[key].properties),
                    bias=self._biases[key_i],
                    in_invariant_features=len(in_invariant_block.properties),
                    hidden_layer_widths=self._hidden_layer_widths[key_i],
                    activation_fn=self._activation_fn,
                    bias_nn=self._bias_nn,
                )
            # Add a layer norm if requested
            if self._normalize_invariants[key_i]:
                assert key["spherical_harmonics_l"] == 0
                block_model = _Sequential(
                    _LayerNorm(len(self._in_metadata[key].properties)),
                    block_model,
                )
            tmp_models.append(block_model)
        self._models = torch.nn.ModuleList(tmp_models)

    @property
    def invariant_baseline(self) -> TensorMap:
        return self._invariant_baseline

    def _set_invariant_baseline(self, invariant_baseline: TensorMap) -> None:
        """
        Sets the output invariant means TensorMap, and stores it in the
        "invariant_baseline" attribute.
        """
        if invariant_baseline is None:
            self._invariant_baseline = None
        else:
            invariant_baseline = invariant_baseline.to(arrays="torch")
            invariant_baseline = invariant_baseline.to(
                dtype=self._torch_settings.get("dtype"),
                device=self._torch_settings.get("device"),
            )
            self._invariant_baseline = metatensor.requires_grad(
                invariant_baseline, False
            )

    @property
    def global_species(self) -> List[int]:
        return self._global_species

    def _set_global_species(self, global_species: List[int]) -> None:
        """
        Sets the global species.
        """
        if global_species is None:
            global_species = np.sort(
                np.unique(self._in_metadata.keys.column("species_center"))
            )
        self._global_species = global_species

    @property
    def descriptor_kwargs(self) -> dict:
        return self._descriptor_kwargs

    def _set_descriptor_kwargs(self, descriptor_kwargs: dict) -> None:
        """
        Sets the kwargs needed for calling the function
        :py:func:`descriptor_builder`.
        """
        if descriptor_kwargs is not None:
            # Also required for descriptor generation are the global species.
            descriptor_kwargs.update({"global_species": self._global_species})
        self._descriptor_kwargs = descriptor_kwargs

    def set_descriptor_kwargs(self, descriptor_kwargs: dict) -> None:
        """
        Sets the kwargs needed for calling the function
        :py:func:`descriptor_builder`.
        """
        self._set_descriptor_kwargs(descriptor_kwargs)

    def update_descriptor_kwargs(self, descriptor_kwargs: dict) -> None:
        """
        Sets the kwargs needed for calling the function
        :py:func:`target_builder`.
        """
        if self._descriptor_kwargs is not None:
            tmp_descriptor_kwargs = self._descriptor_kwargs.copy()
            tmp_descriptor_kwargs.update(descriptor_kwargs)
        else:
            tmp_descriptor_kwargs = descriptor_kwargs
        self._set_descriptor_kwargs(tmp_descriptor_kwargs)

    @property
    def target_kwargs(self) -> dict:
        return self._target_kwargs

    def _set_target_kwargs(self, target_kwargs: dict) -> None:
        """
        Sets the kwargs needed for calling the function
        :py:func:`target_builder`.
        """
        self._target_kwargs = target_kwargs

    def set_target_kwargs(self, target_kwargs: dict) -> None:
        """
        Sets the kwargs needed for calling the function
        :py:func:`target_builder`.
        """
        self._set_target_kwargs(target_kwargs)

    def update_target_kwargs(self, target_kwargs: dict) -> None:
        """
        Sets the kwargs needed for calling the function
        :py:func:`target_builder`.
        """
        if self._target_kwargs is not None:
            tmp_target_kwargs = self._target_kwargs.copy()
            tmp_target_kwargs.update(target_kwargs)
        else:
            tmp_target_kwargs = target_kwargs
        self._set_target_kwargs(tmp_target_kwargs)

    def parameters(self):
        """
        Generator for the parameters of each block model.
        """
        for m in self._models:
            yield from m.parameters()

    def forward(
        self,
        input: Union[TensorMap, List[TensorMap]],
        check_args: bool = True
    ) -> Union[TensorMap, List[TensorMap]]:
        """
        Makes a prediction on an `input` TensorMap or list of TensorMap.

        If the model is trained on standardized outputs, i.e. where the mean
        baseline has been subtracted, this is automatically added back in.
        """
        if isinstance(input, TensorMap):
            input = [input]
            return_list = False
        else:
            assert isinstance(input, List) or isinstance(input, Tuple)
            return_list = True

        predictions = []
        for inp in input:
            # Predict on the keys of the *input* TensorMap
            keys = inp.keys

            # Remove keys that aren't part of the model
            key_mask = torch.tensor(
                [key in self._in_metadata.keys for key in keys], dtype=torch.bool
            )
            if not torch.all(key_mask):
                offending_keys = [key for key in keys if key not in self._in_metadata.keys]
                warnings.warn(
                    f"one or more of input blocks at keys {offending_keys} is not"
                    " part of the keys of the model. The returned prediction will"
                    " not contain these blocks."
                )
            keys = Labels(names=keys.names, values=keys.values[key_mask])

            # Remove the keys for blocks that have no samples
            key_mask = torch.tensor(
                [inp[key].values.shape[0] > 0 for key in keys], dtype=torch.bool
            )
            if not torch.all(key_mask):
                offending_keys = [key for key in keys if inp[key].values.shape[0] == 0]
                warnings.warn(
                    f"one or more of input blocks at keys {offending_keys} has"
                    " zero samples. The returned prediction will not contain"
                    " these blocks."
                )
            keys = Labels(names=keys.names, values=keys.values[key_mask])

            # Make predictions on each block and build the prediction TensorMap
            pred_blocks = []
            for key in keys:
                if check_args:
                    assert key in self._in_metadata.keys
                    if not metatensor.equal_metadata_block(
                        inp[key],
                        self._in_metadata[key],
                        check=["components", "properties"],
                    ):
                        raise ValueError(
                            f"the metadata of the input block at key {key} does not"
                            + " match that of the model"
                        )

                # Get the model
                block_model = self._models[self._in_metadata.keys.position(key)]

                # Make a prediction
                if self._model_type == "linear":
                    pred_values = block_model(inp[key].values, check_args=check_args)
                else:
                    assert self._model_type == "nonlinear"
                    in_invariant = inp.block(
                        spherical_harmonics_l=0, species_center=key["species_center"]
                    )
                    if check_args:
                        assert metatensor.equal_metadata_block(
                            in_invariant,
                            self._in_metadata.block(
                                spherical_harmonics_l=0,
                                species_center=key["species_center"],
                            ),
                            check=["components", "properties"],
                        )
                    pred_values = block_model(
                        inp[key].values,
                        in_invariant=in_invariant.values,
                        check_args=check_args,
                    )

                # Add baseline to invariant blocks if required
                if (
                    self._invariant_baseline is not None
                    and key["spherical_harmonics_l"] == 0
                ):
                    if not isinstance(self._invariant_baseline.block(0), torch.Tensor):
                        self._invariant_baseline = self._invariant_baseline.to(
                            arrays="torch"
                        )
                        self._invariant_baseline = self._invariant_baseline.to(
                            dtype=self._torch_settings["dtype"],
                            device=self._torch_settings["device"],
                        )
                        self._invariant_baseline = metatensor.requires_grad(
                            self._invariant_baseline, False
                        )
                    inv_means = self._invariant_baseline.block(
                        spherical_harmonics_l=0, species_center=key["species_center"]
                    )
                    inv_means_vals = torch.vstack(
                        [inv_means.values for _ in range(pred_values.shape[0])]
                    )
                    pred_values += inv_means_vals

                # Wrap prediction in a TensorBlock and store
                pred_blocks.append(
                    TensorBlock(
                        values=pred_values,
                        samples=inp[key].samples,
                        components=self._out_metadata[key].components,
                        properties=self._out_metadata[key].properties,
                    )
                )

            predictions.append(TensorMap(keys, pred_blocks))

        if len(predictions) == 1 and not return_list:
            return predictions[0]
        return predictions



    def predict(
        self,
        frames: List[ase.Atoms],
        structure_idxs: Optional[List[int]] = None,
        descriptors: Optional[List[TensorMap]] = None,
        build_targets: bool = False,
        return_targets: bool = False,
        save_dir: Optional[Callable] = None,
    ) -> Union[List[TensorMap], List[np.ndarray]]:
        """
        In evaluation mode, makes an end-to-end prediction on a list of ASE
        frames or TensorMap descriptors.

        In either case, the ASE `frames` must be specified.
        """
        self.eval()  # evaluation mode

        # Use continuous structure index range if not specified
        if structure_idxs is None:
            structure_idxs = range(len(frames))
        assert len(structure_idxs) == len(frames)

        with torch.no_grad():
            if descriptors is None:  # from ASE frames
                predictions = self._predict_from_ase(
                    structure_idxs=structure_idxs,
                    frames=frames,
                    build_targets=build_targets,
                    return_targets=return_targets,
                    save_dir=save_dir,
                )

            else:  # from TensorMap descriptors
                assert len(descriptors) == len(frames)
                predictions = self._predict_from_descriptor(
                    frames=frames,
                    structure_idxs=structure_idxs,
                    descriptors=descriptors,
                    build_targets=build_targets,
                    return_targets=return_targets,
                    save_dir=save_dir,
                )

        return predictions

    def _predict_from_ase(
        self,
        structure_idxs: List[int],
        frames: List[ase.Atoms],
        build_targets: bool = False,
        return_targets: bool = False,
        save_dir: Optional[Callable] = None,
    ) -> Union[List[TensorMap], List[np.ndarray]]:
        """
        Makes a prediction on a list of ASE frames.
        """
        if self._descriptor_kwargs is None:
            raise ValueError(
                "if making a prediction on ASE ``frames``,"
                " ``descriptor_kwargs`` must be passed so that"
                " descriptors can be generated. Use the setter"
                " `_set_descriptor_kwargs` to set these and try again."
            )

        # Build the descriptors
        descriptors = predictor.descriptor_builder(
            structure_idxs=structure_idxs,
            frames=frames,
            torch_settings=self._torch_settings,
            **self._descriptor_kwargs,
        )

        # Build predictions from descriptors
        predictions = self._predict_from_descriptor(
            structure_idxs=structure_idxs,
            frames=frames,
            descriptors=descriptors,
            build_targets=build_targets,
            return_targets=return_targets,
            save_dir=save_dir,
        )

        return predictions

    def _predict_from_descriptor(
        self,
        structure_idxs: List[int],
        frames: List[ase.Atoms],
        descriptors: List[TensorMap],
        build_targets: bool = False,
        return_targets: bool = False,
        save_dir: Optional[Callable] = None,
    ) -> Union[List[TensorMap], List[np.ndarray]]:
        """
        Makes a prediction on a list of TensorMap descriptors.
        """
        intermediate_predictions = self(  # predict
            descriptors, check_args=True
        )

        if not build_targets:  # just return coefficients
            return intermediate_predictions

        # Build target
        if self._target_kwargs is None:
            raise ValueError(
                "if ``build_targets`` is true, ``target_kwargs`` must be set"
                " for the model. Use the setter `set_target_kwargs` to do so."
            )
        if save_dir is None:
            raise ValueError(
                "if ``build_targets`` is true, ``save_dir`` must be specified"
            )

        predictions = predictor.target_builder(
            structure_idxs=structure_idxs,
            frames=frames,
            predictions=intermediate_predictions,
            save_dir=save_dir,
            return_targets=return_targets,
            **self._target_kwargs,
        )

        return predictions

class _Sequential(torch.nn.Module):
    def __init__(self, *args):
        super(_Sequential, self).__init__()
        self.layers = torch.nn.ModuleList(args)

    def forward(self, input: torch.Tensor, check_args: bool = True):
        if check_args:
            if not isinstance(input, torch.Tensor):
                raise TypeError("``input`` must be a torch Tensor")
        for layer in self.layers:
            input = layer(input)
        return input


class _LayerNorm(torch.nn.Module):
    def __init__(self, features: int):
        super(_LayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(features)

    def forward(self, input: torch.Tensor, check_args: bool = True):
        if check_args:
            if not isinstance(input, torch.Tensor):
                raise TypeError("``input`` must be a torch Tensor")
        return self.norm(input)


class _LinearModel(torch.nn.Module):
    """
    A linear model, initialized with a number of in and out features (i.e. the
    properties dimension of an metatensor TensorBlock), as well as a bool that
    controls whether or not to use a learnable bias.
    """

    # Initialize model
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(_LinearModel, self).__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, input: torch.Tensor, check_args: bool = True):
        """
        Makes a forward prediction on the ``input`` tensor using linear
        regression.

        If `add_back_inv_means` is true, adds back in the invariant means.
        """
        if check_args:
            if not isinstance(input, torch.Tensor):
                raise TypeError("``input`` must be a torch Tensor")
        return self.linear(input)


class _NonLinearModel(torch.nn.Module):
    """
    A nonlinear torch model. The forward() method takes as input an equivariant
    (i.e. invariant or covariant) torch tensor and an invariant torch tensor.
    The invariant is nonlinearly tranformed by passing it through a sequential
    neural network. The NN architecture is alternating layers of linear and
    nonlinear activation functions. The equivariant block is passed through a
    linear layer before being element-wise multiplied by the invariant output of
    the NN. Then, a this mixed tensor is passed through a linear output layer
    and returned as the prediction.

    This model class must be initialized with several arguments. First, the
    number of ``in_features`` and ``out_features`` of the equivariant block,
    which dictates the widths of the input and output linear layers applied to
    the equivariant.

    Second, the number of features present in the supplementary invariant block,
    ``in_invariant_features`` - this controls the width of the input layer to
    the neural network that the invariant block is passed through.

    Third, the ``hidden_layer_widths`` passed as a list of int. For ``n_elems``
    number of elements in the list, there will be ``n_elems`` number of hidden
    linear layers in the NN architecture, but ``n_elems - 1`` number of
    nonlinear activation layers. Passing a list with 1 element therefore
    corresponds to a linear model, where all equivariant blocks are multiplied
    by their corresponding in_invariant, but with no nonlinearities included.

    Finally, the ``activation_fn`` that should be used must be specified.
    """

    # Initialize model
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        in_invariant_features: int,
        hidden_layer_widths: List[int],
        activation_fn: torch.nn.Module,
        bias_nn: bool = False,
    ):
        super(_NonLinearModel, self).__init__()

        # Define the input layer used on the input equivariant tensor. A
        # learnable bias should only be used if the equivariant passed is
        # invariant, and if requested.
        self.input_layer = torch.nn.Linear(
            in_features=in_features,
            out_features=hidden_layer_widths[-1],
            bias=bias,
        )

        # Define the neural network layers used to nonlinearly transform the
        # invariant tensor. Start with the first linear layer then
        # append pairs of (nonlinear, linear) for each entry in the list of
        # hidden layer widths. As the neural network is only applied to
        # invariants, a learnable bias can be used.
        layers = [
            torch.nn.Linear(
                in_features=in_invariant_features,
                out_features=hidden_layer_widths[0],
                bias=bias_nn,
            )
        ]
        for layer_i in range(0, len(hidden_layer_widths) - 1):
            layers.append(activation_fn)
            layers.append(
                torch.nn.Linear(
                    in_features=hidden_layer_widths[layer_i],
                    out_features=hidden_layer_widths[layer_i + 1],
                    bias=bias_nn,
                )
            )
        self.invariant_nn = torch.nn.Sequential(*layers)

        # Define the output layer that makes the prediction. This acts on
        # equivariants, so should only use a learnable bias if the equivariant
        # passed is an invariant, and if requested.
        self.output_layer = torch.nn.Linear(
            in_features=hidden_layer_widths[-1],
            out_features=out_features,
            bias=bias,
        )

    def forward(
        self,
        input: torch.Tensor,
        in_invariant: torch.Tensor,
        check_args: bool = True,
    ) -> torch.Tensor:
        """
        Makes a forward prediction on the ``input`` tensor that corresponds to
        an equivariant feature. Requires specification of an input invariant
        feature tensor that is passed through a NN and used as a nonlinear
        multiplier to the ``input`` tensor, whilst preserving its equivariant
        behaviour.

        The ``input`` and ``in_invariant`` tensors are torch tensors
        corresponding to i.e. the values of metatensor TensorBlocks. As such,
        they must be 3D tensors, where the first dimension is the samples, the
        last the properties/features, and the 1st (middle) the components. The
        components dimension of the in_invariant block must necessarily be of
        size 1, though that of the equivariant ``input`` can be >= 1, equal to
        (2 \lambda + 1), where \lambda is the spherical harmonic order.

        The ``check_args`` flag can be used to disable the input checking, which
        could be useful for perfomance reasons.
        """
        if check_args:
            # Check inputs are torch tensors
            if not isinstance(input, torch.Tensor):
                raise TypeError("``input`` must be a torch Tensor")
            if not isinstance(in_invariant, torch.Tensor):
                raise TypeError("``in_invariant`` must be a torch Tensor")
            # Check the samples dimensions are the same size between the ``input``
            # equivariant and the ``in_invariant``
            if input.shape[0] != in_invariant.shape[0]:
                raise ValueError(
                    "the samples (1st) dimension of the ``input`` equivariant"
                    + " and the ``in_invariant`` tensors must be equivalent"
                )
            # Check the components (i.e. 2nd) dimension of the in_invariant is 1
            if in_invariant.shape[1] != 1:
                raise ValueError(
                    "the components dimension of the in_invariant block must"
                    + " necessarily be 1"
                )
            # Check the components (i.e. 2nd) dimension of the input equivariant is
            # >= 1 and is odd
            if not (input.shape[1] >= 1 and input.shape[1] % 2 == 1):
                raise ValueError(
                    "the components dimension of the equivariant ``input`` block must"
                    + " necessarily be greater than 1 and odd, corresponding to (2l + 1)"
                )

        # H-stack the in_invariant along the components dimension so that there are
        # (2 \lambda + 1) copies and the dimensions match the equivariant
        in_invariant = torch.hstack([in_invariant] * input.shape[1])

        # Pass the in_invariant tensor through the NN to create a nonlinear
        # multiplier. Also pass the equivariant through a linear input layer.
        nonlinear_multiplier = self.invariant_nn(in_invariant)
        linear_input = self.input_layer(input)

        # Perform element-wise (Hadamard) multiplication of the transformed
        # input with the nonlinear multiplier, which now have the same
        # dimensions
        nonlinear_input = torch.mul(linear_input, nonlinear_multiplier)

        return self.output_layer(nonlinear_input)
