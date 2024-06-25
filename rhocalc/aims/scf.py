import os
from os.path import exists, join
import glob
import shutil

import ase.io

from rhocalc.aims import aims_calc, aims_fields, aims_parser
from rholearn import utils

def set_settings_gloablly(dft_settings: dict):
    """Sets the DFT settings globally."""
    global_vars = globals()
    for key, value in dft_settings.items():
        global_vars[key] = value


def scf(dft_settings: dict):
    """Runs the FHI-aims SCF procedure"""

    set_settings_gloablly(dft_settings)

    for A, frame in zip(SYSTEM_ID, SYSTEM):
        if not exists(SCF_DIR(A)):
            os.makedirs(SCF_DIR(A))
        if not exists(join(SCF_DIR(A), "geometry.in")):
            ase.io.write(join(SCF_DIR(A), "geometry.in"), frame, format="aims")

    # Define paths to the aims.out files for RI calcs
    all_aims_outs = [join(SCF_DIR(A), "aims.out") for A in SYSTEM_ID]
    for aims_out in all_aims_outs:
        if exists(aims_out):
            shutil.copy(aims_out, aims_out + ".previous")
            os.remove(aims_out)

    calcs = {A: {"atoms": structure} for A, structure in zip(SYSTEM_ID, SYSTEM)}
    for A in SYSTEM_ID:
        shutil.copy("dft_settings.py", SCF_DIR(A))
        if exists(join(SCF_DIR(A), "geometry.in.next_step")):
            shutil.copy(
                join(SCF_DIR(A), "geometry.in"),
                join(SCF_DIR(A), "geometry.in.previous"),
            )
            shutil.copy(
                join(SCF_DIR(A), "geometry.in.next_step"),
                join(SCF_DIR(A), "geometry.in"),
            )

        # Define cube settings for Hartree Potential
        if "cube hartree potential" in SCF["output"]:
            com = ALL_SYSTEM[A].get_center_of_mass()
            calcs[A]["aims_kwargs"] = {
                "cubes": (
                    f"cube origin {com[0]} {com[1]} {com[2]} \n"
                    f"cube edge 100 0.1 0.0 0.0 \n"
                    f"cube edge 100 0.0 0.1 0.0 \n"
                    f"cube edge 100 0.0 0.0 0.1 \n"
                )
            }

    # Run the SCF in AIMS
    aims_kwargs = BASE_AIMS.copy()
    aims_kwargs.update(SCF)

    # TODO: parametrize this
    write_geom = True  # False to use optimized geometry instead
    aims_calc.run_aims_array(
        calcs=calcs,
        aims_path=AIMS_PATH,
        aims_kwargs=aims_kwargs,
        sbatch_kwargs=SBATCH,
        run_dir=SCF_DIR,
        load_modules=HPC["load_modules"],
        export_vars=HPC["export_vars"],
        run_command="srun",
        write_geom=write_geom,
    )

def process_scf(dft_settings: dict):
    """
    Parses aims.out file from the SCF procedure. Calculates the Fermi energy by
    integration and stores all the calc info in a dict at `calc_info.pickle`.
    """

    set_settings_gloablly(dft_settings)

    converged = []
    for A, frame in zip(SYSTEM_ID, SYSTEM):
        # Parse the calculation info
        calc_info = aims_parser.parse_aims_out(SCF_DIR(A))
        converged.append(calc_info["scf"]["converged"])

        # Get the Fermi energy as the VBM
        kso_info = aims_parser.get_ks_orbital_info(join(SCF_DIR(A), "ks_orbital_info.out"))
        homo_idx = aims_fields.get_homo_kso_idx(kso_info)
        fermi_vbm = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing

        # Calculate the Fermi energy by integration
        fermi_integrated = None
        # fermi_integrated = aims_fields.calculate_fermi_energy(
        #     kso_info_path=join(SCF_DIR(A), "ks_orbital_info.out"),
        #     n_electrons=frame.get_atomic_numbers().sum(),
        #     gaussian_width=LDOS["gaussian_width"],
        #     interpolation_truncation=9.0,
        # )
        print(f"Fermi energy for {A}: ChemPot: {calc_info['fermi_eV']}, VBM: {fermi_vbm}, integrated: {fermi_integrated}")
        calc_info["vbm_eV"] = fermi_vbm
        calc_info["fermi_integrated_eV"] = fermi_integrated
        utils.pickle_dict(join(SCF_DIR(A), "calc_info.pickle"), calc_info)
        
    assert all(converged)