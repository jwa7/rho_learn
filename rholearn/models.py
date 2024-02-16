"""
Module containing the global nn class `RhoModel`.
"""
from typing import Any, Callable, Dict, List, Optional, Union

import ase
import torch

import metatensor.torch as mts
from metatensor.torch.learn.nn import ModuleMap


from rholearn import predictor


class RhoModel(torch.nn.Module):
    """
    Global model class.

    `nn` is a ModuleMap that acts as the neural network. This can be any network
    of arbitrary architecture.
    """

    def __init__(
        self,
        nn: ModuleMap,
        keys: mts.mts.Labels,
        in_properties: List[mts.Labels],
        out_properties: List[mts.Labels],
        descriptor_calculator: Optional[torch.nn.Module] = None,
        target_kwargs: Optional[Dict[str, Any]] = None,
        **torch_settings,
    ) -> None:

        super().__init__()

        self._nn = nn
        self._torch_settings = torch_settings
        self._set_metadata(input_tensor, output_tensor)
        self._descriptor_kwargs = descriptor_kwargs
        self._target_kwargs = target_kwargs

    def forward(
        self, 
        systems=None,
        descriptors: List[torch.ScriptObject] = None,
        structure_idxs: Optional[List[int]] = None,
        check_metadata: bool = False,
    ) -> List:
        """
        Takes as input either a metatensor
        Calls the forward method of the `self._nn` passed to the constructor,
        but allows for passing a list of inputs.
        """
        with torch.no_grad():

            if systems is not None and descriptors is not None:
                raise ValueError
            if systems is None and descriptors is None:
                raise ValueError

            # Check or generate descriptors
            if systems is not None:  # generate list of descriptors
                assert descriptors is None
                descriptors = self._descriptor_calculator(systems, structure_idxs)
            else:
                if isinstance(descriptors, tuple):
                    descriptors = list(descriptors)
            assert isinstance(descriptors, list)

            # Check the properties metadata
            if check_metadata:
                for desc in descriptors:
                    for key, in_props in zip(self._in_keys, self._in_properties):
                        assert desc[key].properties == in_props

        # Forward
        return [self.__nn(desc) for desc in descriptors]

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
