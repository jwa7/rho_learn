"""
Example settings dict:
```
settings = {
    "io": {
        "data_dir": "/path/to/data/directory",
        "run_dir": "/path/to/simulation/run/directory",
        "coulomb": "/path/to/coulomb_matrices.npz",
    },
    "numpy": {
        "random_seed": 10,
    },
    "train_test_split": {
        "axis": "samples",
        "names": ["structure"],
        "n_groups": 2,
        "group_sizes_rel": [0.8, 0.2],
    },
    "data_partitions": {
        "n_exercises": 1,
        "n_subsets": 1,
    },
    "torch": {
        "requires_grad": True,
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    },
    "model": {
        # "type": "linear",
        "type": "nonlinear_invariants",
        "args": {
            "layer_width": 20,
        },
    },
    "optimizer": {
        "algorithm": torch.optim.LBFGS,
        "args": {
            "lr": 0.2,
        },
    },
    "loss": {
        "fn": "CoulombLoss",
        "args": {
            # "reduction": "sum",
        },
    },
    "training": {
        "n_epochs": 15,
        "save_interval": 3,
    },
}
```
"""
import os
import pprint
from typing import List, Union, Optional

import numpy as np
import torch

import equistore
import equistore.io
from equistore import Labels, TensorMap

from equisolve.utils import split_data

from rholearn import io, loss, models, pretraining, utils


def partition_data(settings: dict):
    """
    Takes as input a dict of ``settings`` and prepares input to model training by
    performing a train-test split and partitioning of the training data into
    subsets of various sizes.

    First, in the parent directory ``settings["io"]["data_dir"]``, the settings
    dict is pickled and saved. Next, a train-test split is performed and the
    resulting TensorMaps are saved to this directory under filenames
    "in_train.npz", "in_test.npz", "out_train.npz", "out_test.npz". The sizes of
    training subsets is also saved here as a numpy array under filename
    "subset_sizes.npy".

    For each learning exercise specified, a subdirectory is created at relative
    path f"exercise_{i}/", where i = (0, ..., n_exercises - 1). The full
    training data (in the parent directory under "in_train.npz" and
    "out_train.npz") is shuffled, and the shuffled structure indices are saved
    to file as a numpy array at relative path "/exercise{i}/sample_indices.npy".

    Then, for each learning exercise, a series of subdirectories are created for
    each training subset, at relative path f"exercise_{i}/subset_{j}/", where j
    = (0, ..., n_subsets - 1). In each of these subdirectories, a subset of the
    full training data is created and saved under the same filenames,
    "in_train.npz" and "out_train.npz".

    Finally, torch global model and loss function objects are created for each
    training subset and saved to the corresponding subset directory.

    The resulting directory structure will be of the form:
    ```
        ``settings["io"]["data_dir"]``/
        |-- settings.pickle
        |-- settings.txt   # just a readable form of the .pickle file
        |-- in_train.npz
        |-- in_test.npz
        |-- out_train.npz
        |-- out_test.npz
        |-- subset_sizes.npy
        |-- exercise_0/
            |-- sample_indices.npy
            |-- subset_0/
                |-- in_train.npz
                |-- out_train.npz
            |-- subset_1/
                |-- in_train.npz
                |-- out_train.npz
            |-- ...
        |-- exercise_1/
            |-- sample_indices.npy
            |-- subset_0/
                |-- in_train.npz
                |-- out_train.npz
            |-- subset_1/
                |-- in_train.npz
                |-- out_train.npz
            |-- ...
        |-- exercise_1/
            ...
    ```
    """
    # Check/create data directory
    io.check_or_create_dir(settings["io"]["data_dir"])

    # Load the input and output data from the paths specified in settings
    print(f"Loading input and output TensorMaps")
    input = equistore.io.load(settings["io"]["input"])
    output = equistore.io.load(settings["io"]["output"])

    # Save settings to pickle and txt files
    print(
        f"Saving data partitioning settings to .txt and .pickle in {settings['io']['data_dir']}"
    )
    io.pickle_dict(
        path=os.path.join(settings["io"]["data_dir"], "settings.pickle"), dict=settings
    )
    with open(os.path.join(settings["io"]["data_dir"], "settings.txt"), "w+") as f:
        f.write(f"Settings:\n\n{pprint.pformat(settings)}")

    # Perform a train-test(-validation) split
    if settings["train_test_split"]["n_groups"] == 2:
        print("Performing train-test split")
    elif settings["train_test_split"]["n_groups"] == 3:
        print("Performing train-test-validation split")
    else:
        raise ValueError(
            "must pass train_test_split n_groups = 2 or 3, to perform a train-test"
            + " or train-test-validation split, respectively"
        )
    tensors, grouped_indices = split_data(
        tensors=[input, output],
        axis="samples",
        names=settings["train_test_split"]["names"],
        n_groups=settings["train_test_split"]["n_groups"],
        group_sizes=settings["train_test_split"].get("group_sizes"),
        seed=settings["numpy"]["random_seed"],
    )

    if settings["train_test_split"]["n_groups"] == 2:
        # Unpack the tensors for the train-test split
        [[in_train, in_test], [out_train, out_test]] = tensors

        # Check that the samples and components indices are exactly equal
        for (a, b) in [(in_train, out_train), (in_test, out_test)]:
            assert utils.equal_metadata(a, b, check=["samples", "components"])

        # Define the filenames to save the structure indices of the partitioned data
        assert len(grouped_indices) == 2
        idx_files = {0: "train", 1: "test", 2: "val"}

        # Define the filenames to save the TensorMaps of the partitioned data
        tm_files = {
            "in_train.npz": in_train,
            "in_test.npz": in_test,
            "out_train.npz": out_train,
            "out_test.npz": out_test,
        }
    else:
        # Unpack the tensors for the train-test-validation split
        [[in_train, in_test, in_val], [out_train, out_test, out_val]] = tensors

        # Check that the samples and components indices are exactly equal
        for (a, b) in [(in_train, out_train), (in_test, out_test), (in_val, out_val)]:
            assert utils.equal_metadata(a, b, check=["samples", "components"])

        # Define the filenames to save the structure indices of the partitioned data
        assert len(grouped_indices) == 3
        idx_files = {0: "train", 1: "test", 2: "val"}

        # Define the filenames to save the TensorMaps of the partitioned data
        tm_files = {
            "in_train.npz": in_train,
            "in_test.npz": in_test,
            "out_train.npz": out_train,
            "out_test.npz": out_test,
            "in_val.npz": in_val,
            "out_val.npz": out_val,
        }

    # Save the structure indices to file
    for i, name in idx_files.items():
        np.save(
            os.path.join(settings["io"]["data_dir"], f"structure_idxs_{name}.npy"),
            grouped_indices[i],
        )

    # Save the TensorMaps to file
    for name, tm in tm_files.items():
        equistore.io.save(os.path.join(settings["io"]["data_dir"], name), tm)

    # Define the train structure indices
    train_structure_idxs = grouped_indices[0]

    # Get train subset sizes (evenly spaced on log base e scale) and write to file
    subset_sizes = utils.get_log_subset_sizes(
        n_max=len(train_structure_idxs),
        n_subsets=settings["data_partitions"]["n_subsets"],
        base=np.e,
    )
    np.save(
        os.path.join(settings["io"]["data_dir"], "subset_sizes_train.npy"),
        subset_sizes,
    )

    # Create training subsets. Increment the random seed for each exercise so
    # that the samples in the training subsets are different.
    for exercise_i in range(settings["data_partitions"]["n_exercises"]):
        # Randomly shuffle the unique indices
        if settings["numpy"]["random_seed"] is not None:
            if settings["numpy"]["random_seed"] != -1:
                # Set a numpy random seed, different for each exercise
                np.random.seed(settings["numpy"]["random_seed"] + exercise_i)
            np.random.shuffle(train_structure_idxs)
        # Create the directories and training data for each subset
        print(f"Creating training subsets for learning exercise {exercise_i}")
        pretraining.create_learning_subsets(
            in_train=in_train,
            out_train=out_train,
            subset_sizes=subset_sizes,
            train_structure_idxs=train_structure_idxs,
            save_dir=os.path.join(settings["io"]["data_dir"], f"exercise_{exercise_i}"),
        )


def create_learning_subsets(
    in_train: TensorMap,
    out_train: TensorMap,
    subset_sizes: np.array,
    train_structure_idxs: Labels,
    save_dir: str,
):
    """
    Takes 2 TensorMaps corresponding to the input and output training data.
    According to the ordered Labels object ``train_structure_idxs`` which
    contains unique sample indices across the TensorMaps (i.e. corresponding to
    unique structure indices), and the list of ``subset_sizes``, slices the
    training data into subsets of various sizes.

    The passed ``train_structure_idxs`` are saved as a numpy array at relative
    path f"{save_dir}/idxs.npy", and the input and output training subsets at
    f"{save_dir}/subset_{i}/in_train.npz" and f".../out_train.npz",
    respectively.
    """
    # Save the (shuffled) sample indices and subset sizes to file
    io.check_or_create_dir(save_dir)
    np.save(os.path.join(save_dir, "structure_idxs_train.npy"), train_structure_idxs)
    # Partition the training data and save the resulting TensorMaps to file
    for subset_i, n_train in enumerate(subset_sizes):
        # Create save directory for this training subset
        subset_save_dir = os.path.join(save_dir, f"subset_{subset_i}")
        io.check_or_create_dir(subset_save_dir)
        # Define the input and output training subset used for this exercise
        in_train_subset = equistore.slice(
            tensor=in_train,
            samples=train_structure_idxs[:n_train],
        )
        out_train_subset = equistore.slice(
            tensor=out_train,
            samples=train_structure_idxs[:n_train],
        )
        # Save in_train, out_train to file
        equistore.io.save(
            os.path.join(subset_save_dir, "in_train.npz"), in_train_subset
        )
        equistore.io.save(
            os.path.join(subset_save_dir, "out_train.npz"), out_train_subset
        )


def construct_torch_objects(settings: dict):
    """
    For each of the training subsets constructs model and loss torch objects.
    Saves them to file in a directory structure mirroring those found in
    settings["io"]["data_dir"], but instead in settings["io"]["run_dir"], i.e.
    with the subdirectory structure <run_dir>/exercise_i/subset_j.
    """
    # Create run dir if not exists
    io.check_or_create_dir(settings["io"]["run_dir"])

    # Save settings to pickle and txt files
    print(
        f"Saving training settings to .txt and .pickle in {settings['io']['run_dir']}"
    )
    io.pickle_dict(
        path=os.path.join(settings["io"]["run_dir"], "settings.pickle"), dict=settings
    )
    with open(os.path.join(settings["io"]["run_dir"], "settings.txt"), "w+") as f:
        f.write(f"Settings:\n\n{pprint.pformat(settings)}")

    # IMPORTANT: set the torch default dtype
    torch.set_default_dtype(settings["torch"]["dtype"])

    # Create test loss fn if using CoulombLoss
    if settings["loss"]["fn"] == "CoulombLoss":
        out_test = equistore.io.load(
            os.path.join(settings["io"]["data_dir"], "out_test.npz")
        )
        loss_fn_test = _init_coulomb_loss_fn(settings, output_like=out_test)

    # Construct model and loss (if using CoulombLoss) objects for every train
    # subdir
    print("Building and saving torch objects in directory:")
    for exercise_i in range(settings["data_partitions"]["n_exercises"]):
        # Create a dir for this exercise
        io.check_or_create_dir(
            os.path.join(settings["io"]["run_dir"], f"exercise_{exercise_i}")
        )
        for subset_j in range(settings["data_partitions"]["n_subsets"]):
            subset_rel_dir = os.path.join(
                f"exercise_{exercise_i}", f"subset_{subset_j}"
            )

            construct_torch_objects_in_train_dir(
                settings,
                data_dir=os.path.join(settings["io"]["data_dir"], subset_rel_dir),
                run_dir=os.path.join(settings["io"]["run_dir"], subset_rel_dir),
            )

            # # Load (to torch) input and ouput data from the data directory
            # in_train = io.load_tensormap_to_torch(
            #     os.path.join(
            #         settings["io"]["data_dir"], subset_rel_dir, "in_train.npz"
            #     ),
            #     **settings["torch"],
            # )
            # out_train = io.load_tensormap_to_torch(
            #     os.path.join(
            #         settings["io"]["data_dir"], subset_rel_dir, "out_train.npz"
            #     ),
            #     **settings["torch"],
            # )

            # # Create a dir for this subset
            # run_dir = os.path.join(settings["io"]["run_dir"], subset_rel_dir)
            # io.check_or_create_dir(run_dir)
            # print(run_dir)

            # # Define some args for initializing the model
            # keys = in_train.keys
            # in_invariant_features = None
            # if settings["model"]["type"] == "nonlinear":
            #     # If using a nonlinear model, the size of the
            #     # properties/features dimension of the nonlinear invariant
            #     # multipliers needs to passed to EquiModelGlobal
            #     invariants = {
            #         specie: in_train.block(
            #             spherical_harmonics_l=0, species_center=specie
            #         )
            #         for specie in np.unique(in_train.keys["species_center"])
            #     }
            #     in_invariant_features = {
            #         key: len(invariants[key[1]].properties) for key in in_train.keys
            #     }
            # # Create model and save
            # model = models.EquiModelGlobal(
            #     model_type=settings["model"]["type"],
            #     keys=keys,
            #     in_feature_labels={key: in_train[key].properties for key in keys},
            #     out_feature_labels={key: out_train[key].properties for key in keys},
            #     in_invariant_features=in_invariant_features,
            #     **settings["model"]["args"],
            # )
            # io.save_torch_object(
            #     torch_obj=model,
            #     path=os.path.join(run_dir, "model.pt"),
            #     torch_obj_str="model",
            # )

            # # If using CoulombLoss: create train loss, save train and test loss
            # if settings["loss"]["fn"] == "CoulombLoss":
            #     loss_fn = _init_coulomb_loss_fn(settings, output_like=out_train)
            #     io.save_torch_object(
            #         torch_obj=loss_fn,
            #         path=os.path.join(run_dir, "loss_fn.pt"),
            #         torch_obj_str="loss_fn",
            #     )
            #     io.save_torch_object(
            #         torch_obj=loss_fn_test,
            #         path=os.path.join(run_dir, "loss_fn_test.pt"),
            #         torch_obj_str="loss_fn",
            #     )


def construct_torch_objects_in_train_dir(settings: dict, data_dir: str, run_dir: str):
    """
    Loads training data from the data directory `data_dir` and constructs
    relevant torch objects, saving them to the run directory `run_dir`.
    """
    # Load (to torch) input and ouput data from the data directory
    in_train = io.load_tensormap_to_torch(
        os.path.join(data_dir, "in_train.npz"),
        **settings["torch"],
    )
    out_train = io.load_tensormap_to_torch(
        os.path.join(data_dir, "out_train.npz"),
        **settings["torch"],
    )

    # Create a dir for this subset
    io.check_or_create_dir(run_dir)
    print(run_dir)

    # Define some args for initializing the model
    keys = in_train.keys
    in_invariant_features = None
    if settings["model"]["type"] == "nonlinear":
        # If using a nonlinear model, the size of the
        # properties/features dimension of the nonlinear invariant
        # multipliers needs to passed to EquiModelGlobal
        invariants = {
            specie: in_train.block(spherical_harmonics_l=0, species_center=specie)
            for specie in np.unique(in_train.keys["species_center"])
        }
        in_invariant_features = {
            key: len(invariants[key[1]].properties) for key in in_train.keys
        }
    # Create model and save
    model = models.EquiModelGlobal(
        model_type=settings["model"]["type"],
        keys=keys,
        in_feature_labels={key: in_train[key].properties for key in keys},
        out_feature_labels={key: out_train[key].properties for key in keys},
        in_invariant_features=in_invariant_features,
        **settings["model"]["args"],
    )
    io.save_torch_object(
        torch_obj=model,
        path=os.path.join(run_dir, "model.pt"),
        torch_obj_str="model",
    )

    # If using CoulombLoss: create train loss, save train and test loss
    if settings["loss"]["fn"] == "CoulombLoss":
        loss_fn = _init_coulomb_loss_fn(settings, output_like=out_train)
        io.save_torch_object(
            torch_obj=loss_fn,
            path=os.path.join(run_dir, "loss_fn.pt"),
            torch_obj_str="loss_fn",
        )
        io.save_torch_object(
            torch_obj=loss_fn_test,
            path=os.path.join(run_dir, "loss_fn_test.pt"),
            torch_obj_str="loss_fn",
        )


def load_training_objects(
    settings: dict,
    train_rel_dir: str,
    restart: Optional[int] = None,
) -> list:
    """
    Returns in the input and output data, the model, loss function, and
    optimizer, either by loading them from file if present, or constructing
    them.

    From the relative training subdirectory `train_rel_dir` loads the torch objects
    "model.pt", and "loss_fn.pt" and "optimizer.pt" if present. If not present,
    constructs them. If using CoulombLoss, "loss_fn_test.pt" is also
    loaded/constructed. If ``restart`` is specified, the "model.pt" and
    "optimizer.pt" are loaded from the indicated epoch checkpoint directory,
    i.e. "train_rel_dir/epoch_{``restart``}/" so that simulations can be continued
    from where they were left off.

    :param settings: a dict of settings
    :param train_rel_dir: the relative training subdirectory. Data is loaded from
        this directory, relative to path ``settings["io"]["data_dir"]``, whilst
        training objects are loaded/saved to the path relative to
        ``settings["io"]["run_dir"]``.
    :param restart: the epoch to restart from. If specified, the model and
        optimizer are loaded from this specified epoch checkpoint directory.

    :return tensors: a list of [in_train, in_test, out_train, out_test], i.e.
        the split training and test data
    :return model: the torch model
    :return loss_fn: the torch loss function
    :return optimizer: the torch optimizer
    """
    # IMPORTANT: set the torch default dtype
    torch.set_default_dtype(settings["torch"]["dtype"])

    # Define the data and run training directories
    train_data_dir = os.path.join(settings["io"]["data_dir"], train_rel_dir)
    train_run_dir = os.path.join(settings["io"]["run_dir"], train_rel_dir)

    # Load input and output train and test data
    in_train = equistore.io.load(os.path.join(train_data_dir, "in_train.npz"))
    out_train = equistore.io.load(os.path.join(train_data_dir, "out_train.npz"))
    in_test = equistore.io.load(os.path.join(settings["io"]["data_dir"], "in_test.npz"))
    out_test = equistore.io.load(
        os.path.join(settings["io"]["data_dir"], "out_test.npz")
    )

    # Standardize the invariant blocks of out_train and out_test
    if settings["training"]["standardize_invariant_features"]:
        # Calculate the means of the invariant features
        train_inv_means = utils.get_invariant_means(out_train)
        # Standardize the train and test data with the train means
        out_train = utils.standardize_invariants(out_train, train_inv_means)
        out_test = utils.standardize_invariants(out_test, train_inv_means)
        # Save the invariant means to file
        equistore.io.save(
            os.path.join(train_data_dir, "inv_means.npz"), train_inv_means
        )

    # Convert the tensors to torch
    in_train = utils.tensor_to_torch(in_train, **settings["torch"])
    in_test = utils.tensor_to_torch(in_test, **settings["torch"])
    out_train = utils.tensor_to_torch(out_train, **settings["torch"])
    out_test = utils.tensor_to_torch(out_test, **settings["torch"])

    # Load model
    model_path = (
        os.path.join(train_run_dir, "model.pt")
        if restart is None
        else os.path.join(train_run_dir, f"epoch_{restart}", "model.pt")
    )

    if os.path.exists(model_path):
        model = io.load_torch_object(
            path=model_path, device=settings["torch"]["device"], torch_obj_str="model"
        )
    else:
        raise FileNotFoundError(
            f"torch model not found at {model_path}. It must be pre-initialized and"
            + " stored in the training directory"
        )
    # Load the loss functions, if present
    loss_fn_path = os.path.join(train_run_dir, "loss_fn.pt")
    if os.path.exists(loss_fn_path):
        loss_fn = io.load_torch_object(
            path=loss_fn_path,
            device=settings["torch"]["device"],
            torch_obj_str="loss_fn",
        )
        loss_fn_test_path = os.path.join(train_run_dir, "loss_fn_test.pt")
        if os.path.exists(loss_fn_path):
            loss_fn_test = io.load_torch_object(
                path=loss_fn_test_path,
                device=settings["torch"]["device"],
                torch_obj_str="loss_fn",
            )
            # If the test loss function exists, compile the train and test losses
            # into a dict
            loss_fn = {"train": loss_fn, "test": loss_fn_test}
    # Otherwise, initialize
    else:
        if settings["loss"]["fn"] == "CoulombLoss":
            # Build train loss fn
            loss_fn_train = _init_coulomb_loss_fn(
                settings=settings, output_like=out_train
            )
            # Build test loss fn
            out_test = equistore.io.load(
                os.path.join(settings["io"]["data_dir"], "out_test.pt")
            )
            loss_fn_test = _init_coulomb_loss_fn(
                settings=settings, output_like=out_test
            )
            loss_fn = {"train": loss_fn_train, "test": loss_fn_test}
        elif settings["loss"]["fn"] == "MSELoss":
            loss_fn = loss.MSELoss(reduction=settings["loss"]["args"]["reduction"])
        else:
            raise NotImplementedError(
                "only CoulombLoss and MSELoss functions currently implemented."
            )

    # Load/create the optimizer
    opt_state_dict_path = (
        os.path.join(train_run_dir, "optimizer.pt")
        if restart is None
        else os.path.join(train_run_dir, f"epoch_{restart}", "optimizer.pt")
    )
    if os.path.exists(opt_state_dict_path):
        opt_state_dict = torch.load(opt_state_dict_path)
        optimizer = torch.optim.LBFGS(model.parameters())
        optimizer.load_state_dict(opt_state_dict)
    else:
        optimizer = settings["optimizer"]["algorithm"](
            params=model.parameters(),
            lr=settings["optimizer"]["args"]["lr"],
        )

    return [in_train, in_test, out_train, out_test], model, loss_fn, optimizer


def _init_coulomb_loss_fn(
    settings: dict,
    output_like: TensorMap,
):
    """
    According to the user settings, initializes the CoulombLoss class and
    returns it.
    """
    # IMPORTANT: set the torch default dtype
    torch.set_default_dtype(settings["torch"]["dtype"])

    # Coulomb matrices file path needs to be specified if using CoulombLoss
    if settings["io"].get("coulomb") is None:
        raise ValueError(
            "must specify filepath for the coulomb matrices in ``settings`` if using CoulombLoss"
        )
    # Check coulomb matrices file path exists
    if not os.path.exists(settings["io"]["coulomb"]):
        raise FileNotFoundError(
            f"coulomb matrices at path {settings['io']['coulomb']} do not exist"
        )
    # Load coulomb matrices
    coulomb_matrices = io.load_tensormap_to_torch(
        os.path.join(settings["io"]["coulomb"]),
        **settings["torch"],
    )
    # Build Coulomb loss function
    loss_fn = loss.CoulombLoss(
        coulomb_matrices=coulomb_matrices,
        output_like=output_like,
    )
    return loss_fn
