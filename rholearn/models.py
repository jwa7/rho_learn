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

    If `out_train_inv_means` is passed as a TensorMap corresponding to the
    invariant means of the output training data, these will be added back to the
    appropriate block in the forward method. As such, the parameters of the
    individual block models will be predicting the baselined coefficients, but
    optimized on the total (unbaselined) coefficients.

    Assumes that, if passed, `out_train_inv_means` is a torch-based TensorMap
    with the same dtype and device as the model, and with `requires_grad=False`.
    """

    # Initialize model
    def __init__(
        self,
        model_type: str,
        input: TensorMap,
        output: TensorMap,
        bias_invariants: bool,
        hidden_layer_widths: Optional[Union[List[List[int]], List[int]]] = None,
        activation_fn: Optional[torch.nn.Module] = None,
        out_train_inv_means: Optional[TensorMap] = None,
        descriptor_kwargs: Optional[dict] = None,
        target_kwargs: Optional[dict] = None,
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

        # Set attributes specific to a nonlinear model
        if self._model_type == "nonlinear":
            self._set_hidden_layer_widths(hidden_layer_widths)
            self._set_activation_fn(activation_fn)

        # Passing `out_train_inv_means` as a TensorMap will add the training
        # invariant means to the relevant block predictions in the `forward`
        # method.
        self._set_out_train_inv_means(out_train_inv_means)

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
                block_model = _LinearModel(
                    in_features=len(self._in_metadata[key].properties),
                    out_features=len(self._out_metadata[key].properties),
                    bias=self._biases[key_i],
                )

            else:
                assert self._model_type == "nonlinear"
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
                )
            tmp_models.append(block_model)
        self._models = torch.nn.ModuleList(tmp_models)

    @property
    def out_train_inv_means(self) -> TensorMap:
        return self._out_train_inv_means

    def _set_out_train_inv_means(self, out_train_inv_means: TensorMap) -> None:
        """
        Sets the output invariant means TensorMap, and stores it in the
        "out_train_inv_means" attribute.
        """
        if out_train_inv_means is None:
            self._out_train_inv_means = None
        else:
            self._out_train_inv_means = metatensor.to(
                out_train_inv_means,
                "torch",
                dtype=self._torch_settings.get("dtype"),
                device=self._torch_settings.get("device"),
                requires_grad=False,
            )

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
            descriptor_kwargs.update(
                {
                    "global_species": np.sort(
                        np.unique(self._in_metadata.keys.column("species_center"))
                    )
                }
            )
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

    def forward(self, input: TensorMap, check_args: bool = True) -> TensorMap:
        """
        Makes a prediction on an `input` TensorMap.

        If the model is trained on standardized outputs, i.e. where the mean
        baseline has been subtracted, this is automatically added back in.
        """
        # Predict on the keys of the *input* TensorMap
        keys = input.keys
        key_mask = torch.tensor(
            [key in self._in_metadata.keys for key in keys], dtype=torch.bool
        )
        if not torch.all(key_mask):
            offending_keys = [key for key in keys if key not in self._in_metadata.keys]
            warnings.warn(
                f"one or more of input blocks at keys {offending_keys} is not "
                " part of the keys of the model. The returned prediction will "
                "not contain this block."
            )
        keys = Labels(names=keys.names, values=keys.values[key_mask])
        pred_blocks = []
        for key in keys:
            if check_args:
                assert key in self._in_metadata.keys
                assert metatensor.equal_metadata_block(
                    input[key],
                    self._in_metadata[key],
                    check=["components", "properties"],
                )

            # Get the model
            block_model = self._models[self._in_metadata.keys.position(key)]

            # Make a prediction
            if self._model_type == "linear":
                pred_values = block_model(input[key].values, check_args=check_args)
            else:
                assert self._model_type == "nonlinear"
                in_invariant = input.block(
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
                    input[key].values,
                    in_invariant=in_invariant.values,
                    check_args=check_args,
                )

            # Add back in the invariant means of the training data to invariant blocks
            if (
                self._out_train_inv_means is not None
                and key["spherical_harmonics_l"] == 0
            ):
                if not isinstance(self._out_train_inv_means.block(0), torch.Tensor):
                    self._out_train_inv_means = metatensor.to(
                        self._out_train_inv_means,
                        backend="torch",
                        requires_grad=False,
                        dtype=self._torch_settings["dtype"],
                        device=self._torch_settings["device"],
                    )
                inv_means = self._out_train_inv_means.block(
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
                    samples=input[key].samples,
                    components=self._out_metadata[key].components,
                    properties=self._out_metadata[key].properties,
                )
            )

        return TensorMap(keys, pred_blocks)

    def predict(
        self,
        structure_idxs: List[int],
        frames: List[ase.Atoms],
        descriptor: Optional[TensorMap] = None,
        build_target: bool = True,
        save_dir: Optional[Callable] = None,
    ) -> TensorMap:
        """
        Performs inference with no gradient tracking to make a prediction on an
        input TensorMap or list of ASE Atoms objects.

        In the case of the former, the descriptor TensorMap is assumed to have
        been generated with the same rascaline hypers as the data the model was
        trained on. In the latter case, the store rascaline hypers are used to
        generate a descriptor for which a prediction is made.

        If `build_target` is false, a list of TensorMaps of predictions for each
        structure in `frames` is returned. If true, a 2-element tuple containing
        a list of prediction TensorMaps and a list of targets for each structure
        in `frames` is returned.

        :param build_target: bool. If true, uses the `predictor.target_builder`
            function along with the `target_kwargs` attribute of the model to
            transform the prediction TensorMap into the desired target. This
            may, for instance, involve calling an external QChem code.
        :param save_dir: callable that returns the directory to save each
            prediction in, taking a single argument corresponding to the
            structure index. Only required if `build_target` is true.
        """
        # Check args
        if build_target:
            if self._target_kwargs is None:
                raise ValueError(
                    "if ``build_target`` is true, ``target_kwargs`` must be set"
                    " for the model. Use the setter `set_target_kwargs` to do so."
                )
            if save_dir is None:
                raise ValueError(
                    "if ``build_target`` is true, ``save_dir`` must be specified"
                )

        # If the equivariant descriptor `input` is not specified, generate a
        # descriptor from the ASE frames
        if descriptor is None:
            if self._descriptor_kwargs is None:
                raise ValueError(
                    "if making a prediction on ASE ``frames``,"
                    " ``descriptor_kwargs`` must be passed so that a"
                    " descriptor can be generated. Use the setter"
                    " `_set_descriptor_kwargs` to set these and try again."
                )

            # Build the descriptor
            descriptor = predictor.descriptor_builder(
                frames,
                torch_settings=self._torch_settings,
                **self._descriptor_kwargs,
            )

            # The structure indices in the descriptor TensorMap will be 0, 1,
            # ..., N_frames by default, according to the order of the structures
            # passed in frames. These will need to be reindexed to match those
            # in `structure_idxs` later.
            actual_structure_idxs = np.arange(len(frames))

        # Check the specified descriptor
        else:
            if not isinstance(descriptor, TensorMap):
                raise TypeError("``descriptor`` must be a TensorMap")

            # Check the structure indices
            tmp_stucture_idxs = metatensor.unique_metadata(
                descriptor, "samples", "structure"
            ).values.reshape(-1)

            err_msg = (
                f"structure indices found in ``descriptor`` ({tmp_stucture_idxs})"
                f" do not match those passed in ``structure_idxs`` ({structure_idxs})."
            )
            if not np.all(np.sort(tmp_stucture_idxs) == np.sort(structure_idxs)):
                raise ValueError(err_msg)

            # The actual structure indices present in the descriptor are the
            # correct ones so will not need modification
            actual_structure_idxs = structure_idxs

        # Make prediction with the model
        with torch.no_grad():

            prediction = self(descriptor, check_args=True)

            # Split the prediction TensorMap by structure index
            predictions = []
            for A, actual_A in zip(structure_idxs, actual_structure_idxs):
                # Split the TensorMap based on the actual structure index present
                tmp_pred = metatensor.slice(
                    prediction,
                    axis="samples",
                    labels=Labels(
                        names=["structure"], 
                        values=np.array([actual_A]).reshape(-1, 1)
                    ),
                )
                if actual_A != A:  # reindex the structure
                    tmp_pred = metatensor.remove_dimension(
                        tmp_pred, axis="samples", name="structure"
                    )
                    tmp_pred = metatensor.insert_dimension(
                        tmp_pred,
                        axis="samples",
                        name="structure",
                        values=np.array([A]),
                        index=0,
                    )
                predictions.append(tmp_pred)

            if not build_target:  # just return the predicted TensorMap
                return predictions

            # Now build the target
            targets = predictor.target_builder(
                structure_idxs=structure_idxs,
                frames=frames,
                predictions=predictions,
                save_dir=save_dir,
                **self._target_kwargs,
            )

            return predictions, targets


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
    ):
        super(_NonLinearModel, self).__init__()

        # Define the input layer used on the input equivariant tensor. A
        # learnable bias should only be used if the equivariant passed is
        # invariant.
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
                bias=bias,
            )
        ]
        for layer_i in range(0, len(hidden_layer_widths) - 1):
            layers.append(activation_fn)
            layers.append(
                torch.nn.Linear(
                    in_features=hidden_layer_widths[layer_i],
                    out_features=hidden_layer_widths[layer_i + 1],
                    bias=bias,
                )
            )
        self.invariant_nn = torch.nn.Sequential(*layers)

        # Define the output layer that makes the prediction. This acts on
        # equivariants, so should only use a learnable bias if the equivariant
        # passed is an invariant.
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
