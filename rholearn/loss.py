from typing import List

import numpy as np
import torch

import equistore
from equistore import Labels, TensorBlock, TensorMap
from rholearn import utils

VALID_LOSS_FNS = ["MSELoss", "CoulombLoss"]


class MSELoss(torch.nn.Module):
    """
    A global loss function so takes TensorMaps as arguments. Calculates the MSE
    (i.e. L2) loss between 2 TensorMaps. Built on top of torch's MSELoss
    calculator. For each block in the input and the corresponding block in the
    target TensorMap, the MSELoss is calculated.

    If ``reduction="sum"``, the final global loss is just given by the simple
    sum of these losses for the individual blocks.

    ..math::
        L = \sum_{b \in blocks} \sum_{i \in b} (\tilde{y}_i - y_i)^2

    If ``reduction="mean_by_values"``, the final global loss is calculated by
    summing the losses of each block, then dividing by the total number of
    elements across all blocks.

    ..math::
        L = \frac{1}{\sum_{b \in blocks} n_b}
            \sum_{b \in blocks} (\tilde{y}_i - y_i)^2

    If ``reduction="mean_by_blocks"``, the final global loss is calculated by
    summing the mean losses of each block, then dividing by the number of
    blocks.

    ..math::
        L = \frac{1}{N_{blocks}} \sum_{b \in blocks} \frac{1}{n_b} \sum_{i in b}
        (\tilde{y}_i - y_i)^2
    """

    def __init__(self, reduction: str):
        super(MSELoss, self).__init__()
        MSELoss._check_input_args(reduction)
        self.reduction = reduction

    @staticmethod
    def _check_input_args(reduction: str):
        if not (reduction in ["sum"]):
            raise ValueError("only ``reduction='sum'`` currently supported")

    def forward(
        self,
        input: TensorMap,
        target: TensorMap,
    ):
        """
        Calculates the MSE Loss between 2 TensorMaps, using the reduction method
        specified at class instantiation.

        :param input: a :py:class:`TensorMap` corresponding to the predicted ML
            e-density. Block values must be torch tensors.
        :param target: a :py:class:`TensorMap` corresponding to the target QM
            e-density. Must by definition have the same data structure (i.e. in
            terms of keys, samples, components, properties) as ``input``. Block
            values must be torch tensors.

        :return loss: a :py:class:`torch.Tensor` corresponding to the total loss
            metric.
        """
        # Input checks
        if not isinstance(input, TensorMap):
            raise TypeError(
                "``input`` arg to ``forward`` must be equistore ``TensorMap``"
            )
        if not isinstance(target, TensorMap):
            raise TypeError(
                "``input`` arg to ``forward`` must be equistore ``TensorMap``"
            )

        # Use the "sum" reduction method to initially calculate the loss for
        # each block, then modify according to the chosen reduction method.
        loss_fn = torch.nn.MSELoss(reduction="sum")
        loss = 0
        for key in input.keys:
            loss += loss_fn(input=input[key].values, target=target[key].values)

        return loss


class CoulombLoss(torch.nn.Module):
    """
    A global loss calculator for calculating the global Coulomb loss metric
    between 2 TensorMaps. Note: doesn't yet account for gradients associated
    with the TensorBlocks.

    This class must be initialized by providing the input ``coulomb_matrices``
    TensorMap, as well as an ``output_like`` TensorMap which is equivalent in
    block dimensions and metadata to the TensorMaps that will be passed to the
    ``forward`` function in runtime. Initialization with ``output_like`` allows
    the ``coulomb_matrices`` to be heavily preprocessed, seeign significant
    speed up in the ``forward`` call.

    The 2 TensorMaps passed to ``forward`` must be of the same dimensions (in
    keys and blocks), and correspond to the ``input`` (i.e. ML) and ``target``
    (i.e. QM) electron density, using a TensorMap of Coulomb matrices that
    describes the repulsive interactions between various environments in the
    ``target`` (i.e. QM) e-density.

    The global loss, $L$, is given by summing over the structures, A, in the
    data. Then, for each combination of $i$ (atoms in A), $a$ (chemical
    species), $l$ (spherical harmonics channel channel), $m$ (spherical
    harmonics component), and $n$ (radial channel), the $\Delta{c}$ (i.e.
    coefficient describing the difference between the target QM and predicted ML
    e-density) the product with all other $\Delta{c}$ and corresponding J_{12}
    (i.e. Coulomb)-coefficient is calculated.

    ..math::
        L =
            \sum_{A \in TrS} \sum{i_1 a_1 l_1 m_1 n_1} \sum{i_2 a_2 l_2 m_2 n_2}
                \delta c_{a_1 l_1 m_1 n_1}^{A i_1} \times \delta c_{a_2 l_2 m_2
                n_2}^{A i_2} \times J_{a_1 l_1 m_1 n_1 a_2 l_2 m_2 n_2}^{A i_1
                i_2}
    """

    def __init__(
        self,
        coulomb_matrices: TensorMap,
        output_like: TensorMap,
    ):
        super(CoulombLoss, self).__init__()
        CoulombLoss._check_init_args(coulomb_matrices, output_like)
        self.coulomb_keys = CoulombLoss.get_coulomb_keys(
            coulomb_matrices,
            output_like,
        )
        self.output_keys = output_like.keys
        self.output_samples = CoulombLoss.get_output_samples(output_like)
        self.output_shapes = CoulombLoss.get_output_shapes(output_like)
        self.processed_coulomb = CoulombLoss.process_coulomb_matrices(
            coulomb_matrices,
            self.coulomb_keys,
            self.output_keys.names,
            self.output_samples,
            self.output_shapes,
        )

    @staticmethod
    def _check_init_args(
        coulomb_matrices: TensorMap,
        output_like: TensorMap,
    ):
        # Check types
        if not isinstance(coulomb_matrices, TensorMap):
            raise TypeError(
                "``coulomb_matrices`` must be an equistore ``TensorMap`` object"
            )
        if not isinstance(output_like, TensorMap):
            raise TypeError("``output_like`` must be an equistore ``TensorMap`` object")

        # Check all the keys of ``output_like`` appear in the keys of ``coulomb_matrices``.
        names = output_like.keys.names
        tmp_c_block_keys = set()
        for l1, l2, a1, a2 in coulomb_matrices.keys:
            tmp_c_block_keys.update({(l1, a1)})
            tmp_c_block_keys.update({(l2, a2)})
        for l, a in output_like.keys:
            if (l, a) not in tmp_c_block_keys:
                raise ValueError(
                    f"key ({l}, {a}) in ``output_like`` not in keys of ``coulomb_matrices``"
                )

    @staticmethod
    def get_coulomb_keys(
        coulomb_matrices: TensorMap,
        output_like: TensorMap,
    ) -> Labels:
        """
        Defines the keys of the Coulomb matrices TensorMap that should be kept,
        according to the keys of the ``output_like`` block.
        """
        # Take only combinations (l1, l2, a1, a2) but not (l2, l1, a2, a1) due
        # to the symmetry of the coulomb matrices
        coulomb_keys = []
        for idx, (l1, a1) in enumerate(output_like.keys):
            for l2, a2 in output_like.keys[idx:]:
                coulomb_keys.append([l1, l2, a1, a2])

        # The Labels object needs to be associated with a TensorBlock so they
        # are searchable
        # TODO: remove this once checked OK
        # return utils.searchable_labels(
        #     Labels(names=coulomb_matrices.keys.names, values=np.array(coulomb_keys))
        # )
        return Labels(names=coulomb_matrices.keys.names, values=np.array(coulomb_keys))

    @staticmethod
    def get_output_samples(output_like: TensorMap) -> dict:
        return {key: block.samples for key, block in output_like}

    @staticmethod
    def get_output_shapes(output_like: TensorMap) -> dict:
        return {key: block.values.shape for key, block in output_like}

    @staticmethod
    def process_coulomb_matrices(
        coulomb_matrices: TensorMap,
        coulomb_keys: Labels,
        output_key_names: tuple,
        output_samples: dict,
        output_shapes: dict,
    ) -> dict:
        """
        Takes the full Coulomb matrices and processes them. Slices each block
        along the samples axis, only keeping samples that appear in the output
        blocks. Reshapes these blocks ready for ``torch.tensordot`` operations
        when ``forward()`` is called.
        """
        processed_coulomb = {}
        for key in coulomb_keys:
            # Unpack l and a indices from the key
            (l1, l2, a1, a2) = key

            coulomb_block = coulomb_matrices[key]

            # Define the keys of the corresponding pair of blocks in the
            # output_like tensor
            key_out_block_1 = utils.key_tuple_to_npvoid(
                (l1, a1), names=output_key_names
            )
            key_out_block_2 = utils.key_tuple_to_npvoid(
                (l2, a2), names=output_key_names
            )

            # Get the unique structure indices for each output block in the pair
            structures_1 = np.unique(output_samples[key_out_block_1]["structure"])
            structures_2 = np.unique(output_samples[key_out_block_2]["structure"])

            # Find the structure indices common to both blocks
            shared_structure_idxs = set(structures_1).intersection(set(structures_2))
            if len(shared_structure_idxs) == 0:
                continue

            structures_dict = {}
            for A in shared_structure_idxs:
                # Create a samples filter (i.e. [True, False, ...]), where true
                # corresponds to samples that contain a structure index shared by the
                # pair of output blocks
                samples_filter = coulomb_block.samples["structure"] == A
                # Slice the Coulomb block according to this filter
                sliced_coulomb_block = coulomb_block.values[samples_filter]
                # Define new shape dimensions
                i1, i2 = (
                    np.sum(output_samples[key_out_block_1]["structure"] == A),
                    np.sum(output_samples[key_out_block_2]["structure"] == A),
                )  # sliced samples dimension
                m1, m2 = (
                    output_shapes[key_out_block_1][1],
                    output_shapes[key_out_block_2][1],
                )  # unsliced components (2nd) dimension
                n1, n2 = (
                    output_shapes[key_out_block_1][2],
                    output_shapes[key_out_block_2][2],
                )  # unsliced properties (3rd) dimension
                # Reshape the Coulomb block
                sliced_coulomb_block = sliced_coulomb_block.reshape(
                    i1, i2, m1, m2, n1, n2
                )
                # Permute the dimensions and make contiguous
                # 0 1 2 3 4 5  -->  0 2 4 1 3 5  before
                # a x b y c z  -->  a b c x y z  after
                sliced_coulomb_block = torch.permute(
                    sliced_coulomb_block, (0, 2, 4, 1, 3, 5)
                ).contiguous()
                # Store the sliced and reshaped block in a dict
                structures_dict[A] = sliced_coulomb_block
            processed_coulomb[key] = structures_dict

        return processed_coulomb

    @staticmethod
    def _coulomb_loss_block_pair(
        delta_block1: TensorBlock,
        delta_block2: TensorBlock,
        coulomb_dict: dict,
    ):
        """
        Calculates the coulomb loss metric between two blocks (that each correspond
        to specific values for the angular channel and species, (l1, a1) and (l2,
        a2) respectively), and the coulomb metric that corresponds to the
        electrostatic repulsion between these blocks (i.e. a coulomb block for (l1,
        l2, a1, a2)).

        :param delta_block1: a :py:class:`TensorBlock` corresponding to the delta
            e-density (i.e. input/predicted ML density minus the target QM density)
            for a given angular channel and species (l1, a1). Block values must be
            torch tensors.
        :param delta_block2: a :py:class:`TensorBlock` corresponding to the delta
            e-density (i.e. input ML minus target QM) for a given angular channel
            and species (l2, a2). May or may not be the same block as
            ``rho_block2``. Block values must be torch tensors.
        :param coulomb_block: a :py:class:`TensorBlock` corresponding to the
            repulsive interactions between atomic environments for a given (l1, l2,
            a1, a2) in the target QM e-density. Block values must be torch tensors.

        :return loss: a :py:class:`float` corresponding to the loss metric for the
            input combination of blocks.
        """
        # Calculate the loss for this block pair by summing over structures
        loss = 0
        for A, coulomb_block in coulomb_dict.items():
            # Slice the delta rho blocks by structure
            block1 = delta_block1.values[delta_block1.samples["structure"] == A]
            block2 = delta_block2.values[delta_block2.samples["structure"] == A]
            # Calculate the loss
            loss += torch.tensordot(
                torch.tensordot(block1, coulomb_block, dims=3), block2, dims=3
            )
        return loss

    def forward(self, input: TensorMap, target: TensorMap):
        """
        Calculates the Coulomb loss between 2 TensorMaps.

        :param input: a :py:class:`TensorMap` corresponding to the predicted ML
            e-density. Block values must be torch tensors.
        :param target: a :py:class:`TensorMap` corresponding to the target QM
            e-density. Must by definition have the same data structure (i.e. in
            terms of keys, samples, components, properties) as ``input``. Block
            values must be torch tensors.

        :return loss: a :py:class:`torch.Tensor` corresponding to the total loss
            metric.
        """
        # Input checks
        if not isinstance(input, TensorMap):
            raise TypeError(
                "``input`` arg to ``forward`` must be equistore ``TensorMap``"
            )
        if not isinstance(target, TensorMap):
            raise TypeError(
                "``input`` arg to ``forward`` must be equistore ``TensorMap``"
            )

        # Get delta electron density TensorMap
        delta_rho = equistore.subtract(input, target)

        # Calculate loss
        loss = 0
        for (l1, l2, a1, a2), coulomb_dict in self.processed_coulomb.items():
            delta_block1 = delta_rho.block(spherical_harmonics_l=l1, species_center=a1)
            delta_block2 = delta_rho.block(spherical_harmonics_l=l2, species_center=a2)
            block_loss = CoulombLoss._coulomb_loss_block_pair(
                delta_block1=delta_block1,
                delta_block2=delta_block2,
                coulomb_dict=coulomb_dict,
            )
            if l1 == l2 and a1 == a2:  # diagonal block
                loss += block_loss
            else:  # off-diagonal block
                loss += 2 * block_loss

        return loss
