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


torch.set_default_dtype(torch.float64)


class LambdaSoapCalculator(torch.nn.Module):
    """
    Defines a torchscriptable lambda-SOAP descriptor calculator.

    :param atom_types: List of atomic numbers for the atom types in the system.
        These should match the global species indices for which the model is
        defined, such that any
    """

    def __init__(
        self,
        atom_types: List[int],
        spherical_expansion_hypers: dict,
        density_correlations_hypers: dict,
        use_lode: bool = False,
    ):
        super(LambdaSoapCalculator, self).__init__()
        self._atom_types = atom_types
        self._spherical_expansion_hypers = spherical_expansion_hypers
        self._density_correlations_hypers = density_correlations_hypers
        if use_lode:
            self._spherical_expansion_calculator = LodeSphericalExpansion(**spherical_expansion_hypers)
        else:
            self._spherical_expansion_calculator = SphericalExpansion(**spherical_expansion_hypers)
        self._density_correlations_calculator = DensityCorrelations(**density_correlations_hypers)

    def forward(
        self,
        system,
        *,
        structure_id: List[int] = None,
        spherical_expansion_compute_args: Optional[Dict] = None,
        density_correlations_compute_args: Optional[Dict] = None,
    ) -> List:
        """
        Computes the lambda-SOAP descriptors for the given systems and returns
        them as per-structure TensorMaps.

            1. Build SphericalExpansion
            2. Moves 'species_neighbor' to properties
            3. Compute DensityCorrelations
            4. Split into per-structure TensorMaps
            5. Reindex the structure indices of each TensorMap if
               `structure_id` is passed
        """
        if spherical_expansion_compute_args is None:
            spherical_expansion_compute_args = {}
        if density_correlations_compute_args is None:
            density_correlations_compute_args = {}

        density = self._spherical_expansion_calculator.compute(
            system, **spherical_expansion_compute_args
        )
        density = density.keys_to_properties(
            keys_to_move=mts.Labels(
                names=["species_neighbor"],
                values=torch.tensor(self._atom_types).reshape(-1, 1),
            )
        )
        lsoap = self._density_correlations_calculator.compute(
            density, **density_correlations_compute_args
        )

        # Find the strutcure indices present in the systems. As `selected_samples` can
        # be passed to the `SphericalExpansion.compute()` method, the structure indices
        # may not be continuous from 0 to len(systems) - 1.
        _ids = mts.unique_metadata(lsoap, "samples", "structure").values.flatten()

        # Split into per-structure TensorMaps. Currently, the TensorMaps have
        # structure indices from 0 to len(system) - 1
        lsoap = [
            mts.slice(
                lsoap,
                "samples",
                labels=mts.Labels(
                    names="structure", values=torch.tensor([A]).reshape(-1, 1)
                ),
            )
            for A in _ids
        ]

        # If `structure_id` is not passed, do not reindex the descriptors.
        if structure_id is None:
            return lsoap

        # If `structure_id` is passed and is equivalent to the unique structure
        # indices found with `unique_metadata`, return the descriptors as is
        if torch.all(torch.tensor(structure_id) == _ids):
            return lsoap

        # Otherwise, reindex the structure indices of each TensorMap
        reindexed_lsoap = []
        for A, desc in zip(structure_id, lsoap):
            # Edit the metadata to match the structure index
            desc = mts.remove_dimension(desc, axis="samples", name="structure")
            new_desc = []
            for key, block in desc.items():
                new_desc.append(
                    mts.TensorBlock(
                        values=block.values,
                        samples=block.samples.insert(
                            index=0,
                            name="structure",
                            values=torch.tensor(
                                [A] * len(block.samples), dtype=torch.int32
                            ),
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                )
            new_desc = mts.TensorMap(desc.keys, new_desc)
            # TODO: when metatensor PR #519 is merged
            # new_desc = mts.insert_dimension(
            #     descriptor,
            #     axis="samples",
            #     name="structure",
            #     values=torch.tensor([A]),
            #     index=0,
            # )
            reindexed_lsoap.append(new_desc)

        return reindexed_lsoap


class RhoModel(torch.nn.Module):
    """
    Global model class.

    `nn` is a ModuleMap that acts as the neural network. This can be any network
    of arbitrary architecture.
    """

    def __init__(
        self,
        in_keys: mts.Labels,
        in_properties: List[mts.Labels],
        out_properties: List[mts.Labels],
        nn: ModuleMap,
        descriptor_calculator: Optional[torch.nn.Module] = None,
        target_kwargs: Optional[Dict] = None,
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
        self._target_kwargs = target_kwargs

    def forward(
        self,
        system=None,
        descriptor: List[torch.ScriptObject] = None,
        structure_id: Optional[List[int]] = None,
        check_metadata: bool = False,
    ) -> List:
        """
        Calls the forward method of the `self._nn` passed to the constructor.
        """

        if system is not None and descriptor is not None:
            raise ValueError
        if system is None and descriptor is None:
            raise ValueError

        # Check or generate descriptors
        if system is not None:  # generate list of descriptors
            assert descriptor is None
            descriptor = self._descriptor_calculator(system=system, structure_id=structure_id)
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
        self, structure: List[ase.Atoms] = None, system=None, structure_id: List = None
    ) -> List:
        """
        Makes a prediction on a list of ASE atoms or a rascaline.Systems object.
        """
        self.eval()
        with torch.no_grad():

            if structure is not None and system is not None:
                raise ValueError

            if structure is not None:
                system = rascaline.torch.systems_to_torch(structure)
            return self(system=system, structure_id=structure_id, check_metadata=True)

