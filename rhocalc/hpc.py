import os



def run_code_via_sbatch(code: str, sbatch_kwargs: dict, hpc_kwargs: dict):
    """
    Run a code on a HPC cluster via an sbatch script.
    """
    # Set up run dir
    run_dir = "."

    # Write the code to file
    with open(os.path.join(run_dir, "run_code.py"), "w") as f:
        f.write(code)

    with open("./run_code.sh", "w") as f:

        # Write the header
        f.write("#!/bin/bash\n")

        # Write the sbatch parameters
        for tag, val in sbatch_kwargs.items():
            f.write(f"#SBATCH --{tag}={val}\n")
        f.write("#SBATCH --get-user-env\n")
        f.write("\n\n")

        f.write("# Run the Python command\n")
        f.write("python -u run_code.py > run_code.out\n\n")

    os.system(f"sbatch run_code.sh")