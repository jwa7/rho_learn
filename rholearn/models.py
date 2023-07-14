from typing import Union, Optional, Sequence

import numpy as np
import torch

from equistore import Labels, TensorBlock, TensorMap
from equistore.core.labels import LabelsEntry


VALID_MODEL_TYPES = ["linear", "nonlinear"]
VALID_ACTIVATION_FNS = ["Tanh", "GELU", "SiLU"]

# ===== RhoModel for making predictions on the TensorMap level


class RhoModel(torch.nn.Module):
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
        in_features: Sequence[Labels],
        out_features: Sequence[Labels],
        hidden_layer_widths: Optional[
            Union[Sequence[Sequence[int]], Sequence[int]]
        ] = None,
        activation_fn: Optional[str] = None,
    ):
        super(RhoModel, self).__init__()
        RhoModel._check_init_args(
            model_type,
            keys,
            in_features,
            out_features,
            hidden_layer_widths,
            activation_fn,
        )
        self.model_type = model_type
        self.keys = keys
        self.in_features = in_features
        self.out_features = out_features

        # Assign attributes specific to nonlinear model
        in_invariant_features = None
        if model_type == "nonlinear":
            # Build a list of the input features of the invariant (l=0) block
            # corresponding to each block model
            in_invariant_features = [
                in_features[keys.position([0, key["species_center"]])] for key in keys
            ]

            # Build a list of the hidden layer widths corresponding to each
            # block model. If passed as a list of list then use as is, otherwise
            # if a single list then use these layer widths for all block models.
            if isinstance(hidden_layer_widths, Sequence) and isinstance(
                hidden_layer_widths[0], int
            ):
                hidden_layer_widths = [hidden_layer_widths for k in keys]
            assert isinstance(hidden_layer_widths, Sequence) and isinstance(
                hidden_layer_widths[0], Sequence
            )

            self.in_invariant_features = in_invariant_features
            self.hidden_layer_widths = hidden_layer_widths
            self.activation_fn = activation_fn

        # Initialize list of block models
        self.models = RhoModel.initialize_models(
            model_type,
            keys,
            in_features,
            out_features,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )

    @staticmethod
    def _check_init_args(
        model_type: str,
        keys: Labels,
        in_features: Sequence[Labels],
        out_features: Sequence[Labels],
        hidden_layer_widths: Optional[
            Union[Sequence[Sequence[int]], Sequence[int]]
        ] = None,
        activation_fn: Optional[str] = None,
    ):
        # Check the length of keys labels
        if not (len(keys) == len(in_features) == len(out_features)):
            raise ValueError(
                "``keys``, ``in_features``, and ``out_features`` must have same length"
            )

        # Check in_features and out_features are list of Labels
        if not isinstance(in_features, Sequence):
            if not np.all([isinstance(f, Labels) for f in in_features]):
                raise TypeError("``in_features`` must be a Sequence[Labels]")
        if not isinstance(out_features, Sequence):
            if not np.all([isinstance(f, Labels) for f in out_features]):
                raise TypeError("``out_features`` must be a Sequence[Labels]")

        # Check model type
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"``model_type`` must be one of: {VALID_MODEL_TYPES}")
        if model_type == "nonlinear":
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
            if not isinstance(hidden_layer_widths, Sequence):
                raise TypeError(
                    "``hidden_layer_widths`` must be passed as Sequence[Sequence[int]] or Sequence[int]), "
                    f"got {type(hidden_layer_widths)}"
                )
            for i in hidden_layer_widths:
                if isinstance(i, Sequence):
                    assert np.all([isinstance(j, int) for j in i])
                else:
                    assert isinstance(i, int)

            # Check activation_fn if using nonlinear model
            if not isinstance(activation_fn, str):
                raise TypeError("``activation_fn`` must be passed as a str")
            if activation_fn not in VALID_ACTIVATION_FNS:
                raise ValueError(
                    f"``activation_fn`` must be one of {VALID_ACTIVATION_FNS}"
                )

    @staticmethod
    def initialize_models(
        model_type: str,
        keys: Labels,
        in_features: Sequence[Labels],
        out_features: Sequence[Labels],
        in_invariant_features: Optional[Sequence[Labels]] = None,
        hidden_layer_widths: Optional[Sequence[Sequence[int]]] = None,
        activation_fn: Optional[str] = None,
    ) -> list:
        """
        Builds a list of torch models for each block in the input/output
        TensorMaps, using a linear or nonlinear model depending on the passed
        ``model_type``. For the invariant (lambda=0) blocks, a learnable bias is
        used in the models, but for covariant blocks no bias is applied.
        """
        # Linear base model
        if model_type == "linear":
            models = [
                RhoModelBlock(
                    key=key,
                    model_type=model_type,
                    in_features=in_feat,
                    out_features=out_feat,
                    bias=True if key["spherical_harmonics_l"] == 0 else False,
                )
                for key, in_feat, out_feat in zip(keys, in_features, out_features)
            ]
        # Nonlinear base model
        elif model_type == "nonlinear":
            models = [
                RhoModelBlock(
                    key=key,
                    model_type=model_type,
                    in_features=in_feat,
                    out_features=out_feat,
                    bias=True if key["spherical_harmonics_l"] == 0 else False,
                    in_invariant_features=in_inv_feat,
                    hidden_layer_widths=hidden_layers,
                    activation_fn=activation_fn,
                )
                for key, in_feat, out_feat, in_inv_feat, hidden_layers in zip(
                    keys,
                    in_features,
                    out_features,
                    in_invariant_features,
                    hidden_layer_widths,
                )
            ]
        else:
            raise ValueError(
                "only 'linear' and 'nonlinear' base model types implemented for RhoModel"
            )
        # Return in a ModuleList so that the models are properly registered
        return torch.nn.ModuleList(models)

    def forward(self, input: TensorMap, check_args: bool = True) -> TensorMap:
        """
        Makes a prediction on the ``input`` TensorMap.

        If the base model type is "linear", a simple linear regression of
        ``input`` is performed.

        If the model type is "nonlinear", the invariant blocks of the ``input``
        TensorMap are nonlinearly transformed and used as multipliers for
        linearly transformed equivariant blocks, which are then regressed in a
        final linear output layer.

        The ``check_args`` flag can be used to disable the input checking, which
        could be useful for perfomance reasons.
        """
        # Perform input checks
        if check_args:
            # Check input TensorMap
            if not isinstance(input, TensorMap):
                raise TypeError("``input`` must be an equistore TensorMap")
            if not np.all([input_key in self.keys for input_key in input.keys]):
                raise ValueError(
                    "the keys of the ``input`` TensorMap given to forward() must match"
                    " the keys used to initialize the RhoModel object. Model keys:"
                    f" {self.keys}, input keys: {input.keys}"
                )

            for key, in_feat in zip(self.keys, self.in_features):
                if input[key].properties != in_feat:
                    raise ValueError(
                        "the feature labels of the ``input`` TensorMap given to forward()"
                        " must match the feature labels used to initialize the"
                        f" RhoModel object. For block {key}, model feature labels:"
                        f" {in_feat}; input feature labels: {input[key].properties}"
                    )
        # Linear base model
        if self.model_type == "linear":
            output = TensorMap(
                keys=self.keys,
                blocks=[
                    model(input[key], check_args=check_args)
                    for key, model in zip(self.keys, self.models)
                ],
            )
        # Nonlinear base model
        elif self.model_type == "nonlinear":
            # Store the invariant (\lambda = 0) blocks in a dict, indexed by
            # the unique chemical species present in the ``input`` TensorMap
            invariants = {
                specie: input.block(spherical_harmonics_l=0, species_center=specie)
                for specie in np.unique(input.keys["species_center"])
            }
            # Return prediction TensorMap
            output = TensorMap(
                keys=self.keys,
                blocks=[
                    model(
                        input[key],
                        invariant=invariants[key["species_center"]],
                        check_args=check_args,
                    )
                    for key, model in zip(self.keys, self.models)
                ],
            )
        else:
            raise ValueError(
                "only 'linear' and 'nonlinear' base model types implemented for RhoModel"
            )
        return output

    def parameters(self):
        """
        Generator for the parameters of each model.
        """
        for model in self.models:
            yield from model.parameters()


# ===== EquiLocalModel for making predictions on the TensorBlock level


class RhoModelBlock(torch.nn.Module):
    """
    A local model used to make predictions at the TensorBlock level. This is
    initialized with input and output feature Labels objects, and returns a
    prediction TensorBlock from its ``forward`` method.
    """

    # Initialize model
    def __init__(
        self,
        key: LabelsEntry,
        model_type: str,
        in_features: Labels,
        out_features: Labels,
        bias: bool,
        in_invariant_features: Optional[Labels] = None,
        hidden_layer_widths: Optional[Sequence[int]] = None,
        activation_fn: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super(RhoModelBlock, self).__init__()
        RhoModelBlock._check_init_args(
            key,
            model_type,
            in_features,
            out_features,
            bias,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )
        # Set torch default dtype
        torch.set_default_dtype(dtype)

        # Assign attributes common to all models
        self.key = key
        self.model_type = model_type
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Assign attributes specific to nonlinear model
        if model_type == "nonlinear":
            self.in_invariant_features = in_invariant_features
            self.hidden_layer_widths = hidden_layer_widths
            self.activation_fn = activation_fn

        # Initialize block model
        self.model = RhoModelBlock.initialize_model(
            model_type,
            in_features,
            out_features,
            bias,
            in_invariant_features,
            hidden_layer_widths,
            activation_fn,
        )

    @staticmethod
    def _check_init_args(
        key: LabelsEntry,
        model_type: str,
        in_features: Labels,
        out_features: Labels,
        bias: bool,
        in_invariant_features: Optional[Labels] = None,
        hidden_layer_widths: Optional[Sequence[int]] = None,
        activation_fn: Optional[str] = None,
    ):
        # Check types
        if not isinstance(key, LabelsEntry):
            raise TypeError(
                "``key`` must be passed as an equistore.core.labels.LabelsEntry"
            )
        if not isinstance(in_features, Labels):
            raise TypeError(
                "``in_features`` must be passed as an equistore Labels object"
            )
        if not isinstance(out_features, Labels):
            raise TypeError(
                "``out_features`` must be passed as an equistore Labels object"
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
            if not isinstance(in_invariant_features, Labels):
                raise TypeError(
                    "``in_invariant_features`` must be passed as an"
                    " equistore Labels object"
                )
            # Check hidden_layer_widths
            if hidden_layer_widths is None:
                raise ValueError(
                    "if using a nonlinear model, you must specify the widths"
                    + " ``hidden_layer_widths`` of each hidden layer in the neural network"
                )
            if not isinstance(hidden_layer_widths, Sequence):
                raise TypeError(
                    "``hidden_layer_widths`` must be passed as Sequence[int]"
                )
            assert np.all([isinstance(width, int) for width in hidden_layer_widths])
            # Check activation_fn
            if not isinstance(activation_fn, str):
                raise TypeError("``activation_fn`` must be passed as a str")
            if activation_fn not in VALID_ACTIVATION_FNS:
                raise ValueError(
                    f"``activation_fn`` must be one of {VALID_ACTIVATION_FNS}"
                )

    @staticmethod
    def initialize_model(
        model_type: str,
        in_features: Labels,
        out_features: Labels,
        bias: bool,
        in_invariant_features: Optional[Labels] = None,
        hidden_layer_widths: Optional[Sequence[int]] = None,
        activation_fn: Optional[str] = None,
    ) -> torch.nn.Module:
        """
        Builds and returns a torch model according to the specified model type
        """
        # Linear base model
        if model_type == "linear":
            model = LinearModel(
                in_features=len(in_features),
                out_features=len(out_features),
                bias=bias,
            )
        # Nonlinear base model
        elif model_type == "nonlinear":
            model = NonLinearModel(
                in_features=len(in_features),
                out_features=len(out_features),
                bias=bias,
                in_invariant_features=len(in_invariant_features),
                hidden_layer_widths=hidden_layer_widths,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                "only 'linear' and 'nonlinear' base model types implemented for RhoModelBlock"
            )
        return model

    def forward(
        self,
        input: TensorBlock,
        invariant: Optional[TensorBlock] = None,
        check_args: bool = True,
    ):
        """
        Makes a prediction on the ``input`` TensorBlock, returning a prediction
        TensorBlock.

        If ``model_type`` is ``linear``, then ``invariant`` is ignored. If
        ``model_type`` is ``nonlinear``, then ``invariant`` must be passed as a
        TensorBlock containing the invariant features to use as a nonlinear
        multiplier for the equivariant `input` block.

        The ``check_args`` flag can be used to disable the input checking, which
        could be useful for perfomance reasons.
        """
        if check_args:
            if not isinstance(input, TensorBlock):
                raise TypeError("``input`` must be an equistore TensorBlock")

            if input.properties != self.in_features:
                raise ValueError(
                    "the feature labels of the ``input`` TensorBlock given to forward()"
                    " must match the feature labels used to initialize the"
                    " RhoModelBlock object. Model feature labels:"
                    f"{self.in_features}, input feature labels: {input.properties}"
                )

        if self.model_type == "linear":
            output = self.model(input.values, check_args)
        elif self.model_type == "nonlinear":
            if check_args:
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
                        " RhoModelBlock object. Model in_invariant_features:"
                        f" {self.in_invariant_features}, invariant feature labels:"
                        f" {invariant.properties}"
                    )
            output = self.model(
                input=input.values, invariant=invariant.values, check_args=check_args
            )
        else:
            raise ValueError(f"``model_type`` must be one of: {VALID_MODEL_TYPES}")

        return TensorBlock(
            samples=input.samples,
            components=input.components,
            properties=self.out_features,
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

    def forward(self, input: torch.Tensor, check_args: bool = True):
        """
        Makes a forward prediction on the ``input`` tensor using linear
        regression.
        """
        if check_args:
            if not isinstance(input, torch.Tensor):
                raise TypeError("``input`` must be a torch Tensor")
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
        hidden_layer_widths: Sequence[int],
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
        hidden_layer_widths: Sequence[int],
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

    def forward(
        self, input: torch.Tensor, invariant: torch.Tensor, check_args: bool = True
    ) -> torch.Tensor:
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

        The ``check_args`` flag can be used to disable the input checking, which
        could be useful for perfomance reasons.
        """
        if check_args:
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
