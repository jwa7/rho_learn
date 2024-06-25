"""
Module containing the global net class `RhoModel`.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import ase
import torch

import metatensor.torch as mts
from metatensor.torch.learn.nn import ModuleMap

import rascaline.torch
from rascaline.torch import LodeSphericalExpansion, SphericalExpansion
from rascaline.torch.utils.clebsch_gordan import DensityCorrelations

from rhocalc import convert
from rhocalc.ase import structure_builder
from rholearn import data


class DescriptorCalculator(torch.nn.Module):
    """
    Transforms an ASE frame into a atom-centered density correlation descriptor,
    according to the specified hypers.
    """

    def __init__(
        self,
        spherical_expansion_hypers: dict,
        density_correlations_hypers: dict,
        atom_types: List[int],
        mask_descriptor: bool = False,
        **mask_kwargs,
    ) -> None:
        super().__init__()

        # Store the settings
        self._spherical_expansion_hypers = spherical_expansion_hypers
        self._density_correlations_hypers = density_correlations_hypers
        self._atom_types = atom_types
        self._mask_descriptor = mask_descriptor
        if self._mask_descriptor:
            assert mask_kwargs is not None
            self._mask_kwargs = mask_kwargs
        else:
            self._mask_kwargs = {}

        # Initialize the calculators
        self._spherical_expansion_calculator = SphericalExpansion(
            **spherical_expansion_hypers
        )
        self._density_correlations_calculator = DensityCorrelations(
            **density_correlations_hypers
        )

    def __repr__(self) -> str:
        return (
            f"DescriptorCalculator("
            f"\n\tspherical_expansion_hypers={self._spherical_expansion_hypers},"
            f"\n\tdensity_correlations_hypers={self._density_correlations_hypers},"
            f"\n\tatom_types={self._atom_types},"
            f"\n\tmask_descriptor={self._mask_descriptor},"
            f"\n\tmask_kwargs={dict(**self._mask_kwargs)},"
            "\n)"
        )

    def compute(
        self,
        system,
        selected_samples: mts.Labels,
        drop_empty_blocks: bool,
        target_keys: Optional[mts.Labels] = None,
    ) -> torch.ScriptObject:
        """
        Takes a rascaline system and computes: 1) a spherical expansion, then 2) takes a
        CG tensor product to produce a lambda-SOAP descriptor.

        Explicit sparsity in the global neighbor types is created to ensure consistent
        properties dimensions.

        The systems present in the output TensorMap `descriptor` will be indexed by the
        continuous numeric range 0 .. len(system) - 1. As such, any system index passed
        in `selected_samples` must reflect this range.

        If `drop_empty_blocks` is True, any blocks in the descriptor that have been
        sliced to zero samples will be removed.

        If `target_keys` is passed, the descriptor will be sliced to only include the
        blocks corresponding to the keys in `target_keys`.
        """
        # Compute Spherical Expansion then Lambda-SOAP
        density = self._spherical_expansion_calculator.compute(
            system, selected_samples=selected_samples
        )
        # Move 'neighbor_type' to properties, accounting for neighbor types not
        # necessarily present in the system but present globally
        density = density.keys_to_properties(
            keys_to_move=mts.Labels(
                names=["neighbor_type"],
                values=torch.tensor(self._atom_types).reshape(-1, 1),
            )
        )
        # Compute lambda-SOAP
        descriptor = self._density_correlations_calculator.compute(density)

        # Remove redundant o3_sigma key name
        descriptor = mts.remove_dimension(descriptor, axis="keys", name="o3_sigma")

        keys_to_drop = []
        if target_keys is not None:  # Slice to only include target keys
            for key in descriptor.keys:
                if key not in target_keys:
                    keys_to_drop.append(key)

        if drop_empty_blocks:
            for key, block in descriptor.items():
                if block.values.shape[0] == 0:  # has been sliced to zero samples
                    if key not in keys_to_drop:
                        keys_to_drop.append(key)

        if len(keys_to_drop) > 0:  # Drop empty blocks
            descriptor = mts.drop_blocks(
                descriptor,
                keys=mts.Labels(
                    names=keys_to_drop[0].names,
                    values=torch.tensor([[i for i in k.values] for k in keys_to_drop]),
                ),
            )

        return descriptor

    def forward(
        self,
        system,
        split_by_system: bool = False,
        drop_empty_blocks: bool = False,
        system_id: List[int] = None,
        target_keys: Optional[mts.Labels] = None,
    ) -> List:
        """
        Computes the lambda-SOAP descriptors for the given systems and returns them as
        per-system TensorMaps.

            - Select atom subsets for each system if computing masked descriptors
            - Compute a SphericalExpansion then take CG tensor product
            - Split into per-system TensorMaps and reindex if appropriate
        """
        selected_samples = None
        if self._mask_descriptor:  # select atom subsets for each system
            selected_samples = select_samples_for_masked_learning(
                system=system, **self._mask_kwargs
            )

        # Compute the descriptor
        descriptor = self.compute(
            system,
            selected_samples,
            drop_empty_blocks=drop_empty_blocks,
            target_keys=target_keys,
        )

        # Split into per-system TensorMaps and reindex if appropriate
        if split_by_system:
            descriptor = split_tensor_and_reindex(
                descriptor, n_systems=len(system), system_id=system_id
            )

        return descriptor


class Model(torch.nn.Module):
    """
    Global model class for predicting a target field on an equivariant basis.

    :param descriptor_calculator: a `torch.nn.Module` that computes descriptors from a
        rascaline.systems object for input into a NN.
    :param target_basis: `dict` the basis set definition for the target scalar field,
        with items for "lmax" and "nmax"
    :param net: `Callable`, a callable function containing the NN architecture as a
        `torch.nn.Module`. Has input arguments that correspond only to model metadata,
        i.e. `in_keys`, `invariant_key_idxs`,`in_properties`, and `out_properties`, and
        returns a fully initialized `torch.nn.Module` once called.
    """

    def __init__(
        self,
        descriptor_calculator: torch.nn.Module,
        target_basis: dict,
        net: Callable,
        dtype,
        device,
    ) -> None:
        super().__init__()

        self._descriptor_calculator = descriptor_calculator
        self._target_basis = target_basis
        self._set_metadata()
        self._dtype = dtype
        self._device = device
        self._set_net(net)

    def _set_metadata(self) -> None:
        """
        Using the target property basis set definition and descriptor calculator, sets
        the model's metadata, i.e. attributes `_in_keys`, `_in_properties`, and
        `_out_properties`
        """
        self._in_keys = target_basis_set_to_in_keys(self._target_basis)
        self._in_properties = atom_types_to_descriptor_basis_in_properties(
            self._in_keys,
            self._descriptor_calculator,
        )
        self._out_properties = target_basis_set_to_out_properties(
            self._in_keys,
            self._target_basis,
        )

    def _set_net(self, net: Callable) -> None:
        """
        Initializes the NN by calling the `net` Callable passed to the constructor.
        The NN is initialized with the model metadata set in `_set_metadata`, and the
        torch settings set in the constructor.
        """
        self._net = net(
            in_keys=self._in_keys,
            in_properties=self._in_properties,
            out_properties=self._out_properties,
            dtype=self._dtype,
            device=self._device,
        )

    def _apply_net(
        self,
        descriptor: Union[torch.ScriptObject, List[torch.ScriptObject]],
        check_metadata: bool,
    ) -> Union[torch.ScriptObject, List[torch.ScriptObject]]:
        """
        Takes a descriptor TensorMap or list of TensorMaps and passes it through the NN.
        Returns the output as a TensorMap or list of TensorMap, respectively.

        If `check_metadata` is True, the properties metadata of the descriptor is
        checked against the model properties metadata.
        """
        # Single TensorMap in, single TensorMap out
        if isinstance(descriptor, torch.ScriptObject):
            if check_metadata:  # check the properties metadata
                for key, in_props in zip(self._in_keys, self._in_properties):
                    if key not in descriptor.keys:
                        continue
                    if not descriptor[key].properties == in_props:
                        raise ValueError(
                            "properties not consistent between model and"
                            f" descriptor at key {key}:\n"
                        )
            return self._net(descriptor)

        # List of TensorMaps in, list of TensorMaps out
        output = []
        for desc in descriptor:
            if check_metadata:  # check the properties metadata
                for key, in_props in zip(self._in_keys, self._in_properties):
                    if key not in desc.keys:
                        continue
                    if not desc[key].properties == in_props:
                        raise ValueError(
                            "properties not consistent between model and"
                            f" descriptor at key {key}:\n"
                        )
            output.append(self._net(desc))

        return output

    def __getitem__(self, i: int) -> ModuleMap:
        """
        Gets the i-th module (i.e. corresponding to the i-th key/block) of the NN.
        """
        return self._net.module_map[i]

    def __iter__(self) -> Tuple[mts.LabelsEntry, ModuleMap]:
        """
        Iterates over the model's NN modules, returning the key and block NN in a tuple.
        """
        return iter(zip(self._in_keys, self._net.module_map))

    def __repr__(self) -> str:
        representation = (
            f"Model("
            f"\ndescriptor_calculator=\n\t"
            + str(self._descriptor_calculator).replace("\n", "\n\t\t")
            + f"\ntarget_basis=\n\t{self._target_basis},"
            f"\ndtype=\n\t{self._dtype},"
            f"\ndevice=\n\t{self._device},"
            "\n"
        )
        representation += f"\nnet="
        for key, block_nn in self:
            representation += f"\n\t{key}:"
            representation += f"\n\t" + str(block_nn).replace("\n", "\n\t\t")
        representation += f"\n)"

        return representation

    def forward(
        self,
        system=None,
        descriptor: Union[torch.ScriptObject, List[torch.ScriptObject]] = None,
        system_id: Optional[List[int]] = None,
        check_metadata: bool = False,
        split_and_reindex: bool = False,
    ) -> List:
        """
        Computes the target property for the given system or descriptor.

        If only `system` is passed, the descriptor is computed and passed through the
        NN. Returned is a list of per-system TensorMaps.

        If `descriptor` is passed as well, the descriptor is not re-computed and instead
        just passed through the NN as is. If passed as a single TensorMap
        (torch.ScriptObject), the NN is applied to the whole descriptor. The resulting
        TensorMap is returned as is unless `split_and_reindex` is set to true, in which
        case it is split into per-system TensorMaps and reindexed to have the correct
        system IDs.

        In both cases, the per-system TensorMaps can be reindexed (in terms of their
        system ID metadata) by passing `system_id`. If `system=None`, they will be
        indexed numerically from [0, ..., len(system) - 1].

        If passed as a list of TensorMaps, each is passed through the NN individually
        with no reindexing.

        If `check_metadata` is True, the properties metadata of the descriptor is
        checked against the model properties metadata.
        """
        if descriptor is None:  # calculate descriptor and keep as single TensorMap
            descriptor = self._descriptor_calculator(
                system=system, 
                split_by_system=False,
                drop_empty_blocks=True,
                target_keys=self._in_keys,
            )

        if isinstance(descriptor, torch.ScriptObject):  # pass whole TM through NN
            output = self._apply_net(descriptor, check_metadata)
            if split_and_reindex is False:
                return output
            if system is None:
                raise ValueError(
                    "must pass `system` if reindexing system IDs of output"
                )
            return split_tensor_and_reindex(
                output, n_systems=len(system), system_id=system_id
            )

        # List of descriptors: pass each TM through NN
        if isinstance(descriptor, tuple):
            descriptor = list(descriptor)
        if not isinstance(descriptor, list):
            raise ValueError(
                f"Expected `descriptor` to be TensorMap or List[TensorMap], got {type(descriptor)}"
            )

        return self._apply_net(descriptor, check_metadata)

    def predict(
        self, frames: Union[ase.Atoms, List[ase.Atoms]], system_id: List[int] = None
    ) -> List:
        """
        Makes a prediction on a list of `ase.Atoms` objects.

        If descriptors are masked, this method will unmask the predictions such that
        the dimensions match the full system. This is done by explicitly padding with
        zeros the atomic samples that were originally masked.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(frames, ase.Atoms):
                frames = [frames]
            system = rascaline.torch.systems_to_torch(frames)
            predictions = self(
                system=system, system_id=system_id, check_metadata=True, split_and_reindex=True
            )

            if self._descriptor_calculator._mask_descriptor is False:  # just return
                return predictions

            # Unmask the predictions
            if system_id is None:
                system_id = torch.arange(len(frames))

            unmasked_predictions = []
            for A, frame, masked_tensor in zip(system_id, frames, predictions):
                unmasked_predictions.append(
                    data.unmask_coeff_vector_tensormap(
                        masked_tensor, self._in_keys, self._out_properties, frame, A
                    )
                )

            return unmasked_predictions


# ===== Helper functions =====


def select_samples_for_masked_learning(
    system,
    surface_depth: Optional[float],
    buffer_depth: Optional[float],
) -> mts.Labels:
    """
    Generates a `mts.Labels` object of the samples selected for masked learning.

    1. Passing a global set of frames to, i.e. rascaline.SphericalExpansion, to then
    compute only a subset. This ensures the feature space has the global dimension.
    2. Computing the atom-centered density correlations for a subset of atoms within
    each frame. This is useful for masked learning, i.e. for learning the surfaces of
    slabs.

    If `masked_learning` is true, `surface_depth` and `buffer_depth` must be passed.
    """
    selected_samples = []
    for A, sys in enumerate(system):
        # Partition atoms into S, I, B regions
        idxs_surface, idxs_buffer, idxs_bulk = (
            structure_builder.get_atom_idxs_by_region(sys, surface_depth, buffer_depth)
        )
        # Keep S + I atoms
        for atom_i in list(idxs_surface) + list(idxs_buffer):
            selected_samples.append([A, atom_i])

    return mts.Labels(names=["system", "atom"], values=torch.tensor(selected_samples))


def split_tensor_and_reindex(
    tensor: torch.ScriptObject, n_systems: int, system_id: Optional[List[int]] = None
) -> List:
    """
    Takes a single TensorMap `tensor` and splits it into per-system TensorMaps. Assumes
    the systems present in `tensor` are indexed numerically from [0, ..., n_systems -
    1]. If `system_id` is passed, each the "system" samples metadata is reindexed for
    each system.
    """
    tensor = [
        mts.slice(  # split into per-system TensorMaps
            tensor,
            "samples",
            labels=mts.Labels(names="system", values=torch.tensor([A]).reshape(-1, 1)),
        )
        for A in torch.arange(n_systems)
    ]
    if system_id is None:  # don't reindex
        return tensor

    assert len(tensor) == len(system_id)

    reindexed_tensor = []
    for A, tnsr in zip(system_id, tensor):  # reindex each TensorMap
        new_tnsr = mts.remove_dimension(tnsr, axis="samples", name="system")
        # TODO: follow up on metatensor issue #600
        # new_tnsr = mts.insert_dimension(
        #     new_tnsr, axis="samples", name="system", values=A, index=0
        # )
        new_tnsr_blocks = []
        for key, block in new_tnsr.items():
            new_tnsr_blocks.append(
                mts.TensorBlock(
                    values=block.values,
                    samples=block.samples.insert(
                        index=0,
                        name="system",
                        values=torch.tensor(
                            [A] * len(block.samples), dtype=torch.int32
                        ),
                    ),
                    components=block.components,
                    properties=block.properties,
                )
            )
        new_tnsr = mts.TensorMap(new_tnsr.keys, new_tnsr_blocks)
        reindexed_tensor.append(new_tnsr)

    return reindexed_tensor


def target_basis_set_to_in_keys(basis_set: dict) -> mts.Labels:
    """
    Converts the basis set definition to the set of in_keys on which the model is defined.

    `basis_set` is a dict of the form, i.e.:
        {
            'lmax': {'Si': 3, 'H': 3},
            'nmax': {
                ('Si', 0): 10, ('Si', 1): 10, ('Si', 2): 8,
                ('H', 0): 9, ('H', 1): 7, ('H', 2): 6,
            }
        }

    And returned is a Labels object with the names "o3_lambda" and "center_type" and
    values, extracted from `basis_set["lmax"]`
    """
    return mts.Labels(
        names=["o3_lambda", "center_type"],
        values=torch.tensor(
            [
                [o3_lambda, convert.SYM_TO_NUM[center_symbol]]
                for center_symbol, o3_lambda_max in basis_set["lmax"].items()
                for o3_lambda in range(o3_lambda_max + 1)
            ]
        ),
    )


def target_basis_set_to_out_properties(
    in_keys: mts.Labels,
    basis_set: dict,
) -> List[mts.Labels]:
    """
    Converts the basis set definition to a list of Labels objects corresponding to the
    out properties for each key in `in_keys`. `basis_set` is a dict, see
    `target_basis_set_to_in_keys` for an example.

    Returned is a list of Labels, each of which enumerate the radial channels "n"
    for each combination of o3_lambda and center_type in `in_keys`, extracted from
    `basis_set["nmax"]`
    """
    out_properties = []
    for key in in_keys:
        o3_lambda, center_type = key
        center_symbol = convert.NUM_TO_SYM[center_type]
        out_properties.append(
            mts.Labels(
                names=["n"],
                values=torch.arange(
                    basis_set["nmax"][(center_symbol, o3_lambda)]
                ).reshape(-1, 1),
            )
        )
    return out_properties


def atom_types_to_descriptor_basis_in_properties(
    in_keys: mts.Labels,
    descriptor_calculator: torch.nn.Module,
) -> List[mts.Labels]:
    """
    Builds a dummy ase.Atoms object from the global atom types in
    `descriptor_calculator._atom_types`, computes the descriptor for it, and extracts
    the properties for each block indexed by the keys in `in_keys`.
    """
    atom_types = descriptor_calculator._atom_types
    dummy_system = rascaline.torch.systems_to_torch(
        [
            ase.Atoms(
                [convert.NUM_TO_SYM[center_type] for center_type in atom_types],
                positions=[[0, 0, 10 * i] for i in range(len(atom_types))],
            )
        ]
    )
    descriptor = descriptor_calculator(system=dummy_system)

    return [descriptor[key].properties for key in in_keys]
