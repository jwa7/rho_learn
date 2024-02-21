"""
Module containing the global nn class `RhoModel`.
"""

from typing import Dict, List, Optional, Union

import ase
import torch

import metatensor.torch as mts
from metatensor.torch.learn.nn import ModuleMap

import rascaline.torch
from rascaline.torch import SphericalExpansion
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
    ):
        super(LambdaSoapCalculator, self).__init__()
        self._atom_types = atom_types
        self._spherical_expansion_calculator = SphericalExpansion(**spherical_expansion_hypers)
        self._density_correlations_calculator = DensityCorrelations(**density_correlations_hypers)

    def forward(
        self,
        systems,
        *,
        structure_idxs: List[int] = None,
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
               `structure_idxs` is passed
        """
        if spherical_expansion_compute_args is None:
            spherical_expansion_compute_args = {}
        if density_correlations_compute_args is None:
            density_correlations_compute_args = {}

        density = self._sphex_calculator.compute(
            systems, **spherical_expansion_compute_args
        )
        density = density.keys_to_properties(
            keys_to_move=mts.Labels(
                names=["species_neighbor"],
                values=torch.tensor(self._atom_types).reshape(-1, 1),
            )
        )
        lsoap = self._cg_calculator.compute(
            density, **density_correlations_compute_args
        )

        # Split into per-structure TensorMaps. Currently, the TensorMaps have
        # strutcure indices from 0 to len(systems) - 1
        lsoap = [
            mts.slice(
                lsoap,
                "samples",
                labels=mts.Labels(
                    names="structure", values=torch.tensor([A]).reshape(-1, 1)
                ),
            )
            for A in range(len(systems))
        ]

        # If `structure_idxs` is passed and is not the contrinuous numeric range 0
        # to len(systems) - 1, then re-index the TensorMaps.
        if structure_idxs is None:
            return lsoap

        # Reindex the structure indices of each TensorMap
        reindexed_lsoap = []
        for A, desc in zip(structure_idxs, lsoap):
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
        # with torch.no_grad():

        if system is not None and descriptor is not None:
            raise ValueError
        if system is None and descriptor is None:
            raise ValueError

        # Check or generate descriptors
        if system is not None:  # generate list of descriptors
            assert descriptor is None
            descriptor = self._descriptor_calculator(system, structure_id)
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
                    assert desc[key].properties == in_props

        return [self._nn(desc) for desc in descriptor]

    def predict(
        self, structure: List[ase.Atoms] = None, system=None, structure_id: List = None
    ) -> List:
        """
        Makes a prediction on a list of ASE atoms or a rascaline.Systems object.
        """
        self.eval()
        with torch.no_grad():

            if frame is not None and system is not None:
                raise ValueError

            if frame is not None:
                system = rascaline.torch.systems_from_torch(frame)
            return self(system=system, structure_id=structure_id, check_metadata=True)

    # def predict(
    #     self,
    #     frames: List[ase.Atoms],
    #     structure_idxs: Optional[List[int]] = None,
    #     descriptors: Optional[List[mts.TensorMap]] = None,
    #     build_targets: bool = False,
    #     return_targets: bool = False,
    #     save_dir: Optional[Callable] = None,
    # ) -> Union[List[mts.TensorMap], List[np.ndarray]]:
    #     """
    #     In evaluation mode, makes an end-to-end prediction on a list of ASE
    #     frames or mts.TensorMap descriptors.

    #     In either case, the ASE `frames` must be specified.
    #     """
    #     self.eval()  # evaluation mode

    #     # Use continuous structure index range if not specified
    #     if structure_idxs is None:
    #         structure_idxs = range(len(frames))
    #     assert len(structure_idxs) == len(frames)

    #     with torch.no_grad():
    #         if descriptors is None:  # from ASE frames
    #             predictions = self._predict_from_ase(
    #                 structure_idxs=structure_idxs,
    #                 frames=frames,
    #                 build_targets=build_targets,
    #                 return_targets=return_targets,
    #                 save_dir=save_dir,
    #             )

    #         else:  # from mts.TensorMap descriptors
    #             assert len(descriptors) == len(frames)
    #             predictions = self._predict_from_descriptor(
    #                 frames=frames,
    #                 structure_idxs=structure_idxs,
    #                 descriptors=descriptors,
    #                 build_targets=build_targets,
    #                 return_targets=return_targets,
    #                 save_dir=save_dir,
    #             )

    #     return predictions

    # def _predict_from_ase(
    #     self,
    #     structure_idxs: List[int],
    #     frames: List[ase.Atoms],
    #     build_targets: bool = False,
    #     return_targets: bool = False,
    #     save_dir: Optional[Callable] = None,
    # ) -> Union[List[mts.TensorMap], List[np.ndarray]]:
    #     """
    #     Makes a prediction on a list of ASE frames.
    #     """
    #     if self._descriptor_kwargs is None:
    #         raise ValueError(
    #             "if making a prediction on ASE ``frames``,"
    #             " ``descriptor_kwargs`` must be passed so that"
    #             " descriptors can be generated. Use the setter"
    #             " `_set_descriptor_kwargs` to set these and try again."
    #         )

    #     # Build the descriptors
    #     descriptors = predictor.descriptor_builder(
    #         structure_idxs=structure_idxs,
    #         frames=frames,
    #         torch_settings=self._torch_settings,
    #         **self._descriptor_kwargs,
    #     )

    #     # Build predictions from descriptors
    #     predictions = self._predict_from_descriptor(
    #         structure_idxs=structure_idxs,
    #         frames=frames,
    #         descriptors=descriptors,
    #         build_targets=build_targets,
    #         return_targets=return_targets,
    #         save_dir=save_dir,
    #     )

    #     return predictions

    # def _predict_from_descriptor(
    #     self,
    #     structure_idxs: List[int],
    #     frames: List[ase.Atoms],
    #     descriptors: List[mts.TensorMap],
    #     build_targets: bool = False,
    #     return_targets: bool = False,
    #     save_dir: Optional[Callable] = None,
    # ) -> Union[List[mts.TensorMap], List[np.ndarray]]:
    #     """
    #     Makes a prediction on a list of mts.TensorMap descriptors.
    #     """
    #     intermediate_predictions = self(descriptors, check_metadata=True)

    #     if not build_targets:  # just return coefficients
    #         return intermediate_predictions

    #     # Build target
    #     if self._target_kwargs is None:
    #         raise ValueError(
    #             "if ``build_targets`` is true, ``target_kwargs`` must be set"
    #             " for the nn. Use the setter `set_target_kwargs` to do so."
    #         )
    #     if save_dir is None:
    #         raise ValueError(
    #             "if ``build_targets`` is true, ``save_dir`` must be specified"
    #         )

    #     predictions = predictor.target_builder(
    #         structure_idxs=structure_idxs,
    #         frames=frames,
    #         predictions=intermediate_predictions,
    #         save_dir=save_dir,
    #         return_targets=return_targets,
    #         **self._target_kwargs,
    #     )

    #     return predictions
