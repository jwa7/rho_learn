"""
Module containing the global nn class `RhoModel`.
"""

from typing import Dict, List, Optional, Union

import ase
import torch

import metatensor.torch as mts
from metatensor.torch.learn.nn import ModuleMap

import rascaline.torch
from rascaline.torch import LodeSphericalExpansion, SphericalExpansion
from rascaline.torch.utils.clebsch_gordan import DensityCorrelations

from rholearn import data


torch.set_default_dtype(torch.float64)


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
            f"\n\tmask_kwargs={self._mask_kwargs},"
            "\n)"
        )

    def compute(self, system, selected_samples: Labels) -> torch.ScriptObject:
        """
        Takes a rascaline system and computes: 1) a spherical expansion, then 2) takes a
        CG tensor product to produce a lambda-SOAP descriptor.

        Explicit sparsity in the global neighbor types is created to ensure consistent
        properties dimensions.

        The systems present in the output TensorMap `descriptor` will be indexed by the
        continuous numeric range 0 .. len(system) - 1. As such, any system index passed
        in `selected_samples` must reflect this range.
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

        return descriptor

    def forward(self, system, system_id: List[int] = None) -> List:
        """
        Computes the lambda-SOAP descriptors for the given systems and returns them as
        per-system TensorMaps.

            - Select atom subsets for each system if computing masked descriptors
            - Compute a SphericalExpansion then take CG tensor product
            - Split into per-system TensorMaps and reindex if appropriate
        """
        selected_samples = None
        if self._mask_descriptor:  # select atom subsets for each system
            selected_samples = data.select_samples_for_masked_learning(
                system=system,
                system_id=system_id,
                masked_learning=self._mask_descriptor,
                **self._mask_kwargs,
            )

        # Compute the descriptor
        descriptor = self.compute(system, selected_samples)

        # Split into per-system TensorMaps and reindex if appropriate
        if system_id is not None:
            descriptor = split_descriptor_and_reindex(
                descriptor, n_systems=len(systems), system_id=system_id
            )

        return descriptor


class Model(torch.nn.Module):
    """
    Global model class that combines the descriptor calculator and neural network
    architecture.
    """

    def __init__(
        self,
        in_keys: mts.Labels,
        in_properties: List[mts.Labels],
        out_properties: List[mts.Labels],
        descriptor_calculator: torch.nn.Module,
        nn: ModuleMap,
        **torch_settings,
    ) -> None:

        super().__init__()
        self._nn = nn
        if torch_settings is None:
            self._torch_settings = {"device": "cpu", "dtype": torch.float64}
        else:
            self._torch_settings = torch_settings
        self._in_keys = in_keys
        self._in_properties = in_properties
        self._out_properties = out_properties
        self._descriptor_calculator = descriptor_calculator

    def forward(
        self,
        system=None,
        descriptor: List[torch.ScriptObject] = None,
        system_id: Optional[List[int]] = None,
        check_metadata: bool = False,
    ) -> List:
        """
        Calls the forward method of the `self._nn` passed to the constructor.
        """
        if system is not None and descriptor is not None:
            raise ValueError("cannot pass both `system` and `descriptor`")
        if system is None and descriptor is None:
            raise ValueError("must pass either `system` or `descriptor`")

        if system is not None:  # calculate descriptors
            descriptor = self._descriptor_calculator(system=system, system_id=system_id)
        else:
            if isinstance(descriptor, tuple):
                descriptor = list(descriptor)
        if not isinstance(descriptor, list):
            raise ValueError(
                f"Expected `descriptor` to be list or tuple, got {type(descriptor)}"
            )

        # Check the properties metadata
        if check_metadata:
            for desc in descriptor:
                for key, in_props in zip(self._in_keys, self._in_properties):
                    if not desc[key].properties == in_props:
                        raise ValueError(
                            "properties not consistent between model and"
                            f" descriptor at key {key}:\n"
                        )

        return [self._nn(desc) for desc in descriptor]

    def predict(
        self, system: List[ase.Atoms] = None, system=None, system_id: List = None
    ) -> List:
        """
        Makes a prediction on a list of ASE atoms or a rascaline.Systems object.
        """
        self.eval()
        with torch.no_grad():

            if system is not None and system is not None:
                raise ValueError

            if system is not None:
                system = rascaline.torch.systems_to_torch(system)
            return self(system=system, system_id=system_id, check_metadata=True)


# ===== Helper functions =====


def split_descriptor_and_reindex(
    self, descriptor: torch.ScriptObject, n_systems: int, system_id: List[int]
) -> List:
    """
    Takes a single descriptor TensorMap output from :py:meth:`compute` and splits it
    into per-system TensorMaps. If `system_id` is passed, reindexes the system
    indices of each TensorMap.
    """
    # # Find the unique system indices present in the systems
    # _ids = mts.unique_metadata(descriptor, "samples", "system").values.flatten()
    _ids = torch.arange(n_systems)

    # Split into per-system TensorMaps. Currently, the TensorMaps have
    # system indices from [0, ..., n_systems - 1]
    descriptor = [
        mts.slice(
            descriptor,
            "samples",
            labels=mts.Labels(names="system", values=torch.tensor([A]).reshape(-1, 1)),
        )
        for A in _ids
    ]

    # If `system_id` is passed and is equivalent to the unique system
    # indices found with `unique_metadata`, return the descriptors as is
    if torch.all(torch.tensor(system_id) == _ids):
        return descriptor

    # Otherwise, reindex the system indices of each TensorMap
    reindexed_descriptor = []
    for A, desc in zip(system_id, descriptor):
        # Edit the metadata to match the system index
        new_desc = mts.remove_dimension(desc, axis="samples", name="system")
        # new_desc = []
        # for key, block in desc.items():
        #     new_desc.append(
        #         mts.TensorBlock(
        #             values=block.values,
        #             samples=block.samples.insert(
        #                 index=0,
        #                 name="system",
        #                 values=torch.tensor(
        #                     [A] * len(block.samples), dtype=torch.int32
        #                 ),
        #             ),
        #             components=block.components,
        #             properties=block.properties,
        #         )
        #     )
        # new_desc = mts.TensorMap(desc.keys, new_desc)
        # TODO: when metatensor PR #519 is merged
        new_desc = mts.insert_dimension(
            new_desc,
            axis="samples",
            name="system",
            values=torch.tensor([A]),
            index=0,
        )
        reindexed_descriptor.append(new_desc)

    return reindexed_descriptor
