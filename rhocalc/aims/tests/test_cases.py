import ase.io

# Define parameters that are different for each run
calcs = {
    # ===== WATER =====
    0: {
        "name": "H2O, cluster, serial",
        "atoms": ase.io.read("systems/water_cluster.xyz"),
        "aims_kwargs": {},
        "sbatch_kwargs": {"ntasks-per-node": 1},
    },
    1: {   
        "name": "H2O, cluster, parallel, n_tasks = 5",
        "atoms": ase.io.read("systems/water_cluster.xyz"),
        "aims_kwargs": {},
        "sbatch_kwargs": {"ntasks-per-node": 5},
    },
    2: {
        "name": "H2O, periodic, 1 kpt, serial",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [1, 1, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 1},
    },
    3: {  
        "name": "H2O, periodic, 1 kpt, parallel, n_tasks (5) > n_kpts",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [1, 1, 1], "collect_eigenvectors": False},
        "sbatch_kwargs": {"ntasks-per-node": 5},
    },
    4: {  
        "name": "H2O, periodic, 1 kpt, parallel, n_tasks (5) > n_kpts, COLLECT",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [1, 1, 1], "collect_eigenvectors": True},
        "sbatch_kwargs": {"ntasks-per-node": 5},
    },
    5: {  
        "name": "H2O, periodic, 4 kpt, parallel, n_tasks (3) < n_kpts",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [2, 2, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 3},
    },
    6: {  
        "name": "H2O, periodic, 4 kpt, parallel, n_tasks (3) < n_kpts, COLLECT",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [2, 2, 1], "collect_eigenvectors": True},
        "sbatch_kwargs": {"ntasks-per-node": 3},
    },
    7: {  
        "name": "H2O, periodic, 4 kpt, parallel, n_tasks (10) > n_kpts",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [2, 2, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 10},
    },
    8: {  
        "name": "H2O, periodic, 4 kpt, parallel, n_tasks (10) > n_kpts, COLLECT",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [2, 2, 1], "collect_eigenvectors": True},
        "sbatch_kwargs": {"ntasks-per-node": 10},
    },
    9: {  
        "name": "H2O, periodic, 4 kpt, serial, n_tasks (1) < n_kpts",
        "atoms": ase.io.read("systems/water_periodic.xyz"),
        "aims_kwargs": {"k_grid": [2, 2, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 1},
    },
    10: {  
        "name": "H2O larger cell, periodic, 1 kpt, parallel, n_tasks (20) > n_kpts (1)",
        "atoms": ase.io.read("systems/water_periodic_larger_cell.xyz"),
        "aims_kwargs": {"k_grid": [1, 1, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 20},
    },
    # ===== SILICON =====
    11: {  
        "name": "Si, periodic, 1 kpt, parallel, n_tasks (20) > n_kpts (2)",
        "atoms": ase.io.read("systems/si_periodic.xyz"),
        "aims_kwargs": {"k_grid": [1, 1, 2]},
        "sbatch_kwargs": {"ntasks-per-node": 20},
    },
    # ===== GOLD =====
    # 11: {

    # },
    # ===== Old test cases =====
    # 4: {
    #     "name": "H2O, periodic, 4 kpt, serial",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_kwargs": {"k_grid": [1, 2, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 1, "time": "03:00:00",},
    # },
    # 5: {  
    #     "name": "H2O, periodic, 4 kpt, parallel, 1 < n_tasks < n_kpts",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_kwargs": {"k_grid": [1, 2, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 3},
    # },
    # 6: {  
    #     "name": "H2O, periodic, 4 kpt, parallel, 1 < n_kpts == n_tasks",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 2, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 4},
    # },
    # 7: {  
    #     "name": "H2O, periodic, 4 kpt, parallel, 1 < n_kpts < n_tasks",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 2, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 7},
    # },
    # 8: {  
    #     "name": "H2O, periodic, 4 kpt, parallel, 1 < n_tasks < n_kpts, collect_eigenvectors",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 2, 2], "collect_eigenvectors": True},
    #     "sbatch_kwargs": {"ntasks-per-node": 3},
    # },
    # 9: {  
    #     "name": "H2O, periodic, 4 kpt, parallel, 1 < n_kpts == n_tasks, collect_eigenvectors",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 2, 2], "collect_eigenvectors": True},
    #     "sbatch_kwargs": {"ntasks-per-node": 4},
    # },
    # 10: {  
    #     "name": "H2O, periodic, 4 kpt, parallel, 1 < n_kpts < n_tasks, collect_eigenvectors",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 2, 2], "collect_eigenvectors": True},
    #     "sbatch_kwargs": {"ntasks-per-node": 7},
    # },
    # 11: {  
    #     "name": "H2O, periodic, 4 kpt, serial, larger cell",
    #     "atoms": ase.io.read("systems/water_periodic_larger_cell.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 2, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 1},
    # },
    # 12: {  
    #     "name": "H2O, periodic, 1 kpt, parallel, n_tasks > n_kpts, larger cell",
    #     "atoms": ase.io.read("systems/water_periodic_larger_cell.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 1, 1]},
    #     "sbatch_kwargs": {"ntasks-per-node": 10},
    # },
    # 13: {  
    #     "name": "H2O, periodic, 2 kpt, parallel, 1 < n_tasks == n_kpts",
    #     "atoms": ase.io.read("systems/water_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 1, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 2},
    # },
    # 14: {  
    #     "name": "H2O, periodic, 2 kpt, parallel, 1 < n_tasks == n_kpts, larger cell",
    #     "atoms": ase.io.read("systems/water_periodic_larger_cell.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 1, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 2},
    # },
    # 15: {  
    #     "name": "Si, periodic, 2 kpt, parallel, 1 < n_tasks == n_kpts",
    #     "atoms": ase.io.read("systems/si_periodic.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 1, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 2},
    # },
    # 16: {  
    #     "name": "Si, periodic, 2 kpt, parallel, 1 < n_tasks == n_kpts, larger cell",
    #     "atoms": ase.io.read("systems/si_periodic_larger_cell.xyz"),
    #     "aims_path": aims_path,
    #     "aims_kwargs": {"k_grid": [1, 1, 2]},
    #     "sbatch_kwargs": {"ntasks-per-node": 2},
    # },
}