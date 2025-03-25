"""
These functions calculate orientational entropy from labelled
coordination shells
"""

import waterEntropy.neighbours.HB as HBond
import waterEntropy.neighbours.RAD as RADShell
from waterEntropy.statistics.convariances import Covariances, get_forces_torques
import waterEntropy.statistics.orientations as Orient
from waterEntropy.statistics.vibrations import Vibrations
from waterEntropy.utils.helpers import nested_dict
import waterEntropy.utils.selections as Select


def get_interfacial_water_orient_entropy(system, start: int, end: int, step: int):
    # pylint: disable=too-many-locals
    """
    For a given system, containing the topology and coordinates of molecules,
    find the interfacial water molecules around solutes and calculate their
    orientational entropy if there is a solute atom in the solvent coordination
    shell.

    :param system: mdanalysis instance of atoms in a frame
    :param start: starting frame number
    :param end: end frame number
    :param step: steps between frames
    """
    # don't need to include the frame_solvent_indices dictionary
    frame_solvent_indices = nested_dict()
    # initialise the Covariance class instance to store covariance matrices
    covariances = Covariances()
    vibrations = Vibrations(temperature=298, force_units="kcal")
    # pylint: disable=unused-variable
    for ts in system.trajectory[start:end:step]:
        # initialise the RAD and HB class instances to store shell information
        shells = RADShell.ShellCollection()
        HBs = HBond.HB()
        # 1. find > 1 UA molecules in system, these are the solutes
        resid_list = Select.find_solute_molecules(system)
        solutes = Select.get_selection(system, "resid", resid_list)
        # 2. find the interfacial solvent molecules that are 1 UA in size
        #   and are in the RAD shell of any solute
        solvent_indices = RADShell.find_interfacial_solvent(solutes, system, shells)
        first_shell_solvent = Select.get_selection(system, "index", solvent_indices)
        # 3. iterate through first shell solvent and find their RAD shells,
        #   HBing in the shells and shell labels
        for solvent in first_shell_solvent:
            # print(solvent)
            # 3a. find RAD shell of interfacial solvent
            shell = RADShell.get_RAD_shell(solvent, system, shells)
            # print(">>>", shell, shell.atom_idx, shell.UA_shell, shells)
            # 3b. find HBing in the shell
            HBond.get_shell_HBs(shell, system, HBs, shells)
            # 3c. find RAD shell labels
            shell = RADShell.get_shell_labels(solvent.index, system, shell, shells)
            # print(">>>", shell, shell.atom_idx)
            # 3d. find HB labels
            HBond.get_HB_labels(solvent.index, system, HBs, shells)
            if shell.nearest_nonlike_idx is not None:
                # 3e. populate the labels into a dictionary for stats
                # only if a different atom is in the RAD shell
                nearest = system.atoms[shell.nearest_nonlike_idx]
                nearest_resid = nearest.resid
                nearest_resname = nearest.resname
                Orient.Labels(
                    nearest_resid,
                    nearest_resname,
                    shell.labels,
                    shell.donates_to_labels,
                    shell.accepts_from_labels,
                )
                frame_solvent_indices = save_solvent_indices(
                    ts.frame,
                    shell.atom_idx,
                    nearest_resid,
                    nearest_resname,
                    frame_solvent_indices,
                )
                # 3f. calculate the running average of force and torque
                # covariance matrices
                solvent_molecule = system.atoms[solvent.index].fragment  # get molecule
                get_forces_torques(
                    covariances,
                    solvent_molecule,
                    f"{nearest_resname}_{nearest_resid}",
                    system,
                )
        # 4. clear each shell and HB dictionary ready for the next frame.
        # RADShell.RAD.shells.clear()
        # HBond.HB.donating_to.clear()
        # HBond.HB.accepting_from.clear()

    # 5. get the orientational entropy of interfacial waters and save
    #   them to a dictionary
    # TO-DO: add average Nc in Sorient dict
    Sorient_dict = Orient.get_resid_orientational_entropy_from_dict(
        Orient.Labels.resid_labelled_shell_counts
    )
    # 6. Get the vibrational entropy of interfacial waters
    vibrations.add_data(covariances, diagonalise=True)
    return (
        Sorient_dict,
        covariances,
        vibrations,
        frame_solvent_indices,
    )


def get_solvent_vibrational_entropy(system, frame_solvent_indices):
    # pylint: disable=unused-argument
    """
    This function will be used to get the vibrational entropies of solvent
    molecules collected from orientational entropy calculations. The
    interfacial waters in this dict are the indices of the UA in the water.
    """
    return None


def save_solvent_indices(
    frame: int,
    atom_idx: int,
    nearest_resid: int,
    nearest_resname: str,
    frame_solvent_indices: dict,
):
    """
    Save the solvent indices at interfaces per frame into a dictionary

    :param frame: frame number of analysed frame
    :param atom_idx: solvent atom index
    :param nearest_resid: residue of number of nearest solute molecule
    :param nearest_resname: residue name of nearest solute molecule
    :param frame_solvent_indices: the dictionary to populate
    """
    if nearest_resid not in frame_solvent_indices[frame][nearest_resname]:
        frame_solvent_indices[frame][nearest_resname][nearest_resid] = []
    frame_solvent_indices[frame][nearest_resname][nearest_resid].append(atom_idx)
    return frame_solvent_indices


def print_Sorient_dicts(Sorient_dict: dict):
    """
    Print the orientational entropies of interfacial solvent
    """
    for resid, resname_key in sorted(list(Sorient_dict.items())):
        for resname, [Sor, count] in sorted(list(resname_key.items())):
            print(resid, resname, Sor, count)


def print_frame_solvent_dicts(frame_solvent_indices: dict):
    """
    Print the interfacial solvent for each analysed frame
    """
    for frame, resname_key in sorted(list(frame_solvent_indices.items())):
        for resname, resid_key in sorted(list(resname_key.items())):
            for resid, solvents in sorted(list(resid_key.items())):
                print(frame, resname, resid, len(solvents), solvents)
