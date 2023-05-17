from typing import List, Dict, Union, Optional

import numpy as np
import torch

from equistore import Labels, TensorBlock, TensorMap



VALID_MODEL_TYPES = ["linear", "nonlinear"]
VALID_ACTIVATION_FNS = ["Tanh", "GELU", "SiLU"]

# ===== EquiModelGlobal for making predictions on the TensorMap level


class EquiModelGlobal(torch.nn.Module):
    """
    A single global model that wraps multiple individual models for each block
    of the input TensorMaps. Returns a prediction TensorMap from its ``forward``
    method.
    """

    # Initialize model
    def __init__(
        self,
        model_type: str,
        keys: Labels,
        in_feature_labels: Dict[np.void, Labels],
        out_feature_labels: Dict[np.void, Labels],
        in_invariant_features: Optional[Dict[np.void, int]] = None,
        hidden_layer_widths: Optional[
            Union[Dict[np.void, List[int]], List[int]]
        ] = None,
        activation_fn: Optional[str] = None,
    ):
        super(EquiModelGlobal, self).__init__()
        EquiModelGlobal._check_init_args(
            model_type,
            keys,
            in_feature_labels,
            out_feature_labels,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )
        self.model_type = model_type
        self.keys = keys
        self.in_feature_labels = in_feature_labels
        self.out_feature_labels = out_feature_labels
        if model_type == "nonlinear":
            in_invariant_features = (
                in_invariant_features
                if isinstance(in_invariant_features, dict)
                else {k: in_invariant_features for k in self.keys}
            )
            hidden_layer_widths = (
                hidden_layer_widths
                if isinstance(hidden_layer_widths, dict)
                else {k: hidden_layer_widths for k in self.keys}
            )
            self.in_invariant_features = in_invariant_features
            self.hidden_layer_widths = hidden_layer_widths
            self.activation_fn = activation_fn
        self.models = EquiModelGlobal.build_model_dict(
            model_type,
            keys,
            in_feature_labels,
            out_feature_labels,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )

    @staticmethod
    def _check_init_args(
        model_type: str,
        keys: Labels,
        in_feature_labels: Dict[np.void, Labels],
        out_feature_labels: Dict[np.void, Labels],
        in_invariant_features: Optional[Dict[np.void, int]] = None,
        hidden_layer_widths: Optional[
            Union[Dict[np.void, List[int]], List[int]]
        ] = None,
        activation_fn: Optional[str] = None,
    ):
        # Check the length of keys labels
        if (len(keys) != len(in_feature_labels.keys())) or (
            len(keys) != len(out_feature_labels.keys())
        ):
            raise ValueError(
                "there must be the same number of labels in ``keys`` as there are keys in"
                + " the dict keys of ``in_feature_labels`` and ``out_feature_labels``"
            )

        for key in keys:
            # Check all keys are in the in_feature_labels and out_feature_labels
            # dict keys
            if key not in in_feature_labels.keys():
                raise ValueError(
                    "``keys`` contains a key that isn't present in the dict keys of ``in_feature_labels``"
                )
            if key not in out_feature_labels.keys():
                raise ValueError(
                    "``keys`` contains a key that isn't present in the dict keys of ``out_feature_labels``"
                )
            # Check all values of the dicts are Labels objects
            if not isinstance(in_feature_labels[key], Labels):
                raise TypeError(
                    "values of the ``in_features_labels`` dict must be Labels objects"
                )
            if not isinstance(out_feature_labels[key], Labels):
                raise TypeError(
                    "values of the ``out_feature_labels`` dict must be Labels objects"
                )
        # Check model type
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"``model_type`` must be one of: {VALID_MODEL_TYPES}")
        if model_type == "nonlinear":
            # Check in_invariant_features if using nonlinear model
            if in_invariant_features is None:
                raise ValueError(
                    "if using a nonlinear model, you must specify the number"
                    + " ``in_invariant_features`` of features that the invariant blocks"
                    + " used as nonlinear multipliers for each equivariant block will contain."
                    + " This must be passed as a dict indexed by equivariant block key."
                )
            if not isinstance(in_invariant_features, dict):
                raise TypeError(
                    "``in_invariant_features`` must be passed as a dict of int"
                )
            for v in in_invariant_features.values():
                if not isinstance(v, int):
                    raise TypeError(
                        "each value in ``in_invariant_features`` must be passed as an int"
                    )
            # Check hidden_layer_widths if using nonlinear model
            if hidden_layer_widths is None:
                raise ValueError(
                    "if using a nonlinear model, you must specify the number"
                    + " ``hidden_layer_widths`` of features in the final hidden"
                    + " layer of the NN applied to the invariant blocks, for each block"
                    + " indexed by block key"
                    "if using a nonlinear model, you must specify the widths"
                    + " ``hidden_layer_widths`` of each hidden layer in the neural network"
                )
            if not isinstance(hidden_layer_widths, (dict, list)):
                raise TypeError(
                    "``hidden_layer_widths`` must be passed as a dict of list of int, or a list of int"
                )
            if isinstance(hidden_layer_widths, dict):
                for v1 in hidden_layer_widths.values():
                    if not isinstance(v1, list):
                        raise TypeError(
                            "each value in ``hidden_layer_widths`` must be passed as a list of int"
                        )
                    for v2 in v1:
                        if not isinstance(v2, int):
                            raise TypeError(
                                "each value in ``hidden_layer_widths`` must be passed as an int"
                            )
            if not isinstance(activation_fn, str):
                raise TypeError("``activation_fn`` must be passed as a str")
            if activation_fn not in VALID_ACTIVATION_FNS:
                raise ValueError(
                    f"``activation_fn`` must be one of {VALID_ACTIVATION_FNS}"
                )

    @staticmethod
    def build_model_dict(
        model_type: str,
        keys: Labels,
        in_feature_labels: Dict[np.void, Labels],
        out_feature_labels: Dict[np.void, Labels],
        in_invariant_features: Optional[Dict[np.void, int]] = None,
        hidden_layer_widths: Optional[
            Union[Dict[np.void, List[int]], List[int]]
        ] = None,
        activation_fn: Optional[str] = None,
    ) -> dict:
        """
        Builds a dict of torch models for each block in the input/output
        TensorMaps, using a linear or nonlinear model depending on the passed
        ``model_type``. For the invariant (lambda=0) blocks, a learnable bias is
        used in the models, but for covariant blocks no bias is applied.
        """
        if model_type == "linear":
            models = {
                key: EquiModelLocal(
                    model_type=model_type,
                    in_feature_labels=in_feature_labels[key],
                    out_feature_labels=out_feature_labels[key],
                    bias=True
                    if key[list(keys.names).index("spherical_harmonics_l")] == 0
                    else False,
                )
                for key in keys
            }
        elif model_type == "nonlinear":
            models = {
                key: EquiModelLocal(
                    model_type=model_type,
                    in_feature_labels=in_feature_labels[key],
                    out_feature_labels=out_feature_labels[key],
                    bias=True
                    if key[list(keys.names).index("spherical_harmonics_l")] == 0
                    else False,
                    in_invariant_features=in_invariant_features[key],
                    hidden_layer_widths=hidden_layer_widths[key],
                    activation_fn=activation_fn,
                )
                for key in keys
            }
        else:
            raise ValueError(
                "only 'linear' and 'nonlinear' base model types implemented for EquiModelGlobal"
            )
        return models

    def forward(self, input: TensorMap):
        """
        Makes a prediction on the ``input`` TensorMap. If the base model type is
        "linear", a simple linear regression of ``input`` is performed. If the
        model type is "nonlinear", the invariant blocks of the ``input``
        TensorMap are nonlinearly transformed and used as multipliers for
        linearly transformed equivariant blocks, which are then regressed in a
        final linear output layer.
        """
        if not isinstance(input, TensorMap):
            raise TypeError("``input`` must be an equistore TensorMap")
        if not np.all([input_key in self.keys for input_key in input.keys]):
            raise ValueError(
                "the keys of the ``input`` TensorMap given to forward() must match"
                " the keys used to initialize the EquiModelGlobal object. Model keys:"
                f"{self.keys}, input keys: {input.keys}"
            )

        for key in self.keys:
            if input[key].properties != self.in_feature_labels[key]:
                raise ValueError(
                    "the feature labels of the ``input`` TensorMap given to forward()"
                    " must match the feature labels used to initialize the"
                    f" EquiModelGlobal object. For block {key}, model feature labels:"
                    f"{self.in_feature_labels[key]}, input feature"
                    f" labels: {input[key].properties}"
                )

        if self.model_type == "linear":
            output = TensorMap(
                keys=self.keys,
                blocks=[self.models[key](input[key]) for key in self.keys],
            )
        elif self.model_type == "nonlinear":
            # Store the invariant (\lambda = 0) blocks in a dict, indexed by
            # the unique chemical species present in the ``input`` TensorMap
            invariants = {
                specie: input.block(spherical_harmonics_l=0, species_center=specie)
                for specie in np.unique(input.keys["species_center"])
            }
            # Make a prediction for each block
            pred_blocks = []
            for key in self.keys:
                _, specie = key
                pred_blocks.append(
                    self.models[key](input=input[key], invariant=invariants[specie])
                )
            output = TensorMap(keys=self.keys, blocks=pred_blocks)
        else:
            raise ValueError(
                "only 'linear' and 'nonlinear' base model types implemented for EquiModelGlobal"
            )
        return output

    def parameters(self):
        """
        Generator for the parameters of each model.
        """
        for model in self.models.values():
            yield from model.parameters()


# ===== EquiLocalModel for making predictions on the TensorBlock level


class EquiModelLocal(torch.nn.Module):
    """
    A local model used to make predictions at the TensorBlock level. This is
    initialized with input and output feature Labels objects, and returns a
    prediction TensorBlock from its ``forward`` method.

    If the base torch model type specified is ``model_type="nonlinear"``, then
    ``nn_layer_width`` and ``last_hidden_features`` must also be specified upon
    class initialization. The former controls the width of all except the last
    hidden layers in the neural network used to nonlinearly transform invariant
    features when calling ``forward()``. The latter controls the width of the
    final hidden layer, before the output layer.
    """

    # Initialize model
    def __init__(
        self,
        model_type: str,
        in_feature_labels: Labels,
        out_feature_labels: Labels,
        bias: bool,
        in_invariant_features: Optional[int] = None,
        hidden_layer_widths: Optional[List[int]] = None,
        activation_fn: Optional[str] = None,
    ):
        super(EquiModelLocal, self).__init__()
        EquiModelLocal._check_init_args(
            model_type,
            in_feature_labels,
            out_feature_labels,
            bias,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )
        self.model_type = model_type
        self.in_feature_labels = in_feature_labels
        self.out_feature_labels = out_feature_labels
        self.bias = bias
        if model_type == "nonlinear":
            self.in_invariant_features = in_invariant_features
            self.hidden_layer_widths = hidden_layer_widths
            self.activation_fn = activation_fn
        self.model = EquiModelLocal.build_model(
            model_type,
            in_feature_labels,
            out_feature_labels,
            bias,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )

    @staticmethod
    def _check_init_args(
        model_type: str,
        in_feature_labels: Labels,
        out_feature_labels: Labels,
        bias: bool,
        in_invariant_features: Optional[int] = None,
        hidden_layer_widths: Optional[List[int]] = None,
        activation_fn: Optional[str] = None,
    ):
        # Check types
        if not isinstance(in_feature_labels, Labels):
            raise TypeError(
                "``in_feature_labels`` must be passed as an equistore Labels object"
            )
        if not isinstance(out_feature_labels, Labels):
            raise TypeError(
                "``out_feature_labels`` must be passed as an equistore Labels object"
            )
        if not isinstance(bias, bool):
            raise TypeError("``bias`` must be passed as a bool")
        if not isinstance(model_type, str):
            raise TypeError("``model_type`` must be passed as a str")
        # Check model type
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"``model_type`` must be one of: {VALID_MODEL_TYPES}")
        if model_type == "nonlinear":
            # Check in_invariant_features
            if in_invariant_features is None:
                raise ValueError(
                    "if using a nonlinear model, you must specify the number"
                    + " ``in_invariant_features`` of features that the invariant block"
                    + " will contain"
                )
            if not isinstance(in_invariant_features, int):
                raise TypeError("``in_invariant_features`` must be passed as an int")
            # Check hidden_layer_widths
            if hidden_layer_widths is None:
                raise ValueError(
                    "if using a nonlinear model, you must specify the widths"
                    + " ``hidden_layer_widths`` of each hidden layer in the neural network"
                )
            if not isinstance(hidden_layer_widths, list):
                raise TypeError("``hidden_layer_widths`` must be passed as list of int")
            # Check activation_fn
            if not isinstance(activation_fn, str):
                raise TypeError("``activation_fn`` must be passed as a str")
            if activation_fn not in VALID_ACTIVATION_FNS:
                raise ValueError(
                    f"``activation_fn`` must be one of {VALID_ACTIVATION_FNS}"
                )

    @staticmethod
    def build_model(
        model_type: str,
        in_feature_labels: Labels,
        out_feature_labels: Labels,
        bias: bool,
        in_invariant_features: Optional[int] = None,
        hidden_layer_widths: Optional[List[int]] = None,
        activation_fn: Optional[str] = None,
    ) -> torch.nn.Module:
        """
        Builds and returns a torch model according to the specified model type
        """
        if model_type == "linear":
            model = LinearModel(
                in_features=len(in_feature_labels),
                out_features=len(out_feature_labels),
                bias=bias,
            )
        elif model_type == "nonlinear":
            model = NonLinearModel(
                in_features=len(in_feature_labels),
                out_features=len(out_feature_labels),
                bias=bias,
                in_invariant_features=in_invariant_features,
                hidden_layer_widths=hidden_layer_widths,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                "only 'linear' and 'nonlinear' base model types implemented for EquiModelLocal"
            )
        return model

    def forward(self, input: TensorBlock, invariant: Optional[TensorBlock] = None):
        """
        Makes a prediction on the ``input`` TensorBlock, returning a prediction
        TensorBlock.
        """
        if not isinstance(input, TensorBlock):
            raise TypeError("``input`` must be an equistore TensorBlock")

        if input.properties != self.in_feature_labels:
            raise ValueError(
                "the feature labels of the ``input`` TensorBlock given to forward()"
                " must match the feature labels used to initialize the"
                " EquiModelLocal object. Model feature labels:"
                f"{self.in_feature_labels}, input feature labels: {input.properties}"
            )

        if self.model_type == "linear":
            output = self.model(input.values)
        elif self.model_type == "nonlinear":
            # Check samples exactly equivalent
            if input.samples != invariant.samples:
                raise ValueError(
                    "``input`` and ``invariant`` TensorBlocks must have the"
                    + " the same samples Labels, in the same order."
                )
            # Check input invariant features match that of the model
            if invariant.properties != self.in_invariant_features:
                raise ValueError(
                    "the feature labels of the ``invariant`` TensorBlock given to forward()"
                    " must match the invariant features used to initialize the"
                    " EquiModelLocal object. Model in_invariant_features:"
                    f"{self.in_invariant_features}, invariant feature labels:"
                    f"{invariant.properties}"
                )
            output = self.model(input=input.values, invariant=invariant.values)
        else:
            raise ValueError(f"``model_type`` must be one of: {VALID_MODEL_TYPES}")

        return TensorBlock(
            samples=input.samples,
            components=input.components,
            properties=self.out_feature_labels,
            values=output,
        )

    def parameters(self):
        """
        Generator for the parameters of the model
        """
        return self.model.parameters()


# === Torch models that makes predictions on torch Tensors


class LinearModel(torch.nn.Module):
    """
    A linear model, initialized with a number of in and out features (i.e. the
    properties dimension of an equistore TensorBlock), as well as a bool that
    controls whether or not to use a learnable bias.
    """

    # Initialize model
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(LinearModel, self).__init__()
        LinearModel._check_init_args(in_features, out_features, bias)
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    @staticmethod
    def _check_init_args(in_features: int, out_features: int, bias: bool):
        if not isinstance(in_features, int):
            raise TypeError("``in_features`` must be passed as an int")
        if not isinstance(out_features, int):
            raise TypeError("``out_features`` must be passed as an int")
        if not isinstance(bias, bool):
            raise TypeError("``bias`` must be passed as an bool")

    def forward(self, input: torch.Tensor):
        """
        Makes a forward prediction on the ``input`` tensor using linear
        regression.
        """
        return self.linear(input)


class NonLinearModel(torch.nn.Module):
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
    by their corresponding invariants, but with no nonlinearities included.

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
        activation_fn: str,
    ):
        super(NonLinearModel, self).__init__()
        NonLinearModel._check_init_args(
            in_features,
            out_features,
            bias,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )

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
                bias=True,
            )
        ]
        for layer_i in range(0, len(hidden_layer_widths) - 1):
            if activation_fn == "Tanh":
                layers.append(torch.nn.Tanh())
            elif activation_fn == "GELU":
                layers.append(torch.nn.GELU())
            elif activation_fn == "SiLU":
                layers.append(torch.nn.SiLU())
            else:
                raise ValueError(
                    f"``activation_fn`` must be one of {VALID_ACTIVATION_FNS}"
                )
            layers.append(
                torch.nn.Linear(
                    in_features=hidden_layer_widths[layer_i],
                    out_features=hidden_layer_widths[layer_i + 1],
                    bias=True,
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

    @staticmethod
    def _check_init_args(
        in_features: int,
        out_features: int,
        bias: bool,
        in_invariant_features: int,
        hidden_layer_widths: List[int],
        activation_fn: str,
    ):
        if not isinstance(in_features, int):
            raise TypeError("``in_features`` must be passed as an int")
        if not isinstance(out_features, int):
            raise TypeError("``out_features`` must be passed as an int")
        if not isinstance(bias, bool):
            raise TypeError("``bias`` must be passed as a bool")
        if not isinstance(in_invariant_features, int):
            raise TypeError("``in_invariant_features`` must be passed as an int")
        if not isinstance(hidden_layer_widths, list):
            raise TypeError("``hidden_layer_widths`` must be passed as a list of int")
        for width in hidden_layer_widths:
            if not isinstance(width, int):
                raise TypeError(
                    "``hidden_layer_widths`` must be passed as a list of int"
                )
        if not isinstance(activation_fn, str):
            raise TypeError("``activation_fn`` must be passed as a str")
        if activation_fn not in VALID_ACTIVATION_FNS:
            raise ValueError(f"``activation_fn`` must be one of {VALID_ACTIVATION_FNS}")

    def forward(self, input: torch.Tensor, invariant: torch.Tensor):
        """
        Makes a forward prediction on the ``input`` tensor that corresponds to
        an equivariant feature. Requires specification of an invariant feature
        tensor that is passed through a NN and used as a nonlinear multiplier to
        the ``input`` tensor, whilst preserving its equivariant behaviour.

        The ``input`` and ``invariant`` tensors are torch tensors corresponding
        to i.e. the values of equistore TensorBlocks. As such, they must be 3D
        tensors, where the first dimension is the samples, the last the
        properties/features, and the 1st (middle) the components. The components
        dimension of the invariant block must necessarily be of size 1, though
        that of the equivariant ``input`` can be >= 1, equal to (2 \lambda + 1),
        where \lambda is the spherical harmonic order.
        """
        # Check inputs are torch tensors
        if not isinstance(input, torch.Tensor):
            raise TypeError("``input`` must be a torch Tensor")
        if not isinstance(invariant, torch.Tensor):
            raise TypeError("``invariant`` must be a torch Tensor")
        # Check the samples dimensions are the same size between the ``input``
        # equivariant and the ``invariant``
        if input.shape[0] != invariant.shape[0]:
            raise ValueError(
                "the samples (1st) dimension of the ``input`` equivariant"
                + " and the ``invariant`` tensors must be equivalent"
            )
        # Check the components (i.e. 2nd) dimension of the invariant is 1
        if invariant.shape[1] != 1:
            raise ValueError(
                "the components dimension of the invariant block must"
                + " necessarily be 1"
            )
        # Check the components (i.e. 2nd) dimension of the input equivariant is
        # >= 1 and is odd
        if not (input.shape[1] >= 1 and input.shape[1] % 2 == 1):
            raise ValueError(
                "the components dimension of the equivariant ``input`` block must"
                + " necessarily be greater than 1 and odd, corresponding to (2l + 1)"
            )

        # H-stack the invariant along the components dimension so that there are
        # (2 \lambda + 1) copies and the dimensions match the equivariant
        invariant = torch.hstack([invariant] * input.shape[1])

        # Pass the invariant tensor through the NN to create a nonlinear
        # multiplier. Also pass the equivariant through a linear input layer.
        nonlinear_multiplier = self.invariant_nn(invariant)
        linear_input = self.input_layer(input)

        # Perform element-wise (Hadamard) multiplication of the transformed
        # input with the nonlinear multiplier, which now have the same
        # dimensions
        nonlinear_input = torch.mul(linear_input, nonlinear_multiplier)

        # Finally pass through the output layer and return
        return self.output_layer(nonlinear_input)
