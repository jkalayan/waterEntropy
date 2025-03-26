"""
These functions calculate orientational entropy from labelled
coordination shells
"""

import waterEntropy.analysis.HB as HBond
import waterEntropy.analysis.HB_labels as HBLabels
import waterEntropy.analysis.RAD as RADShell
import waterEntropy.analysis.shell_labels as RADLabels
from waterEntropy.analysis.shells import ShellCollection
from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.entropy.orientations import Orientations
from waterEntropy.entropy.vibrations import Vibrations
from waterEntropy.recipes.forces_torques import get_forces_torques
from waterEntropy.utils.helpers import nested_dict
import waterEntropy.utils.selections as Selections


def find_interfacial_solvent(solutes, system, shells: ShellCollection):
    """
    For a given set of solute molecules, find the RAD shells for each UA in the
    molecules, if a solvent molecule is in the RAD shell, then save the solvent
    atom index to a list. A solvent is defined as molecule that constitutes a
    single UA. These solvent molecule are defined as interfacial molecules.

    :param solutes: mdanalysis instance of a selection of atoms in solute
        molecules that are greater than one UA
    :param system: mdanalysis instance of atoms in a frame
    :param shells: ShellCollection instance
    """
    solvent_indices = []
    molecules = solutes.fragments  # fragments is mdanalysis equiv to molecules
    for molecule in molecules:
        # 1. find heavy atoms in the molecule
        UAs = Selections.find_molecule_UAs(molecule)
        for atom in UAs:
            # 2. find the shell of each UA atom in a molecule
            shell = RADShell.get_RAD_shell(
                atom, system, shells
            )  # get the molecule UA shell
            shell_indices = shell.UA_shell
            # 3. for each neighbour in the RAD shell, find single UA molecules
            shell_atoms = Selections.get_selection(system, "index", shell_indices)
            for neighbour_atom in shell_atoms:
                # create an atom group to find molecule/fragment
                neighbour_atomGroup = Selections.get_selection(
                    system, "index", [neighbour_atom.index]
                )
                neighbour_molecule = neighbour_atomGroup.fragments[0]
                neighbour_UAs = Selections.find_molecule_UAs(
                    neighbour_molecule.fragments[0]
                )  # 1 length list
                if len(neighbour_UAs) == 1:  # single UA molecule is a solvent
                    # 5. add single UA molecules into the solvent indices list
                    if neighbour_atom.index not in solvent_indices:
                        solvent_indices.append(neighbour_atom.index)
                    else:
                        continue
                else:
                    continue
    return solvent_indices


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
    covariances = CovarianceCollection()
    # initialise the Vibrations class instance to store vibrational entropies
    vibrations = Vibrations(temperature=298, force_units="kcal")
    hb_labels = HBLabels.HBLabelCollection()
    # pylint: disable=unused-variable
    for ts in system.trajectory[start:end:step]:
        # initialise the RAD and HB class instances to store shell information
        shells = ShellCollection()
        HBs = HBond.HBCollection()
        # 1. find > 1 UA molecules in system, these are the solutes
        resid_list = Selections.find_solute_molecules(system)
        solutes = Selections.get_selection(system, "resid", resid_list)
        # 2. find the interfacial solvent molecules that are 1 UA in size
        #   and are in the RAD shell of any solute
        solvent_indices = find_interfacial_solvent(solutes, system, shells)
        first_shell_solvent = Selections.get_selection(system, "index", solvent_indices)
        # 3. iterate through first shell solvent and find their RAD shells,
        #   HBing in the shells and shell labels
        for solvent in first_shell_solvent:
            # print(solvent)
            # 3a. find RAD shell of interfacial solvent
            shell = RADShell.get_RAD_shell(solvent, system, shells)
            # 3b. find HBing in the shell
            HBond.get_shell_HBs(shell, system, HBs, shells)
            # 3c. find RAD shell labels
            shell = RADLabels.get_shell_labels(solvent.index, system, shell, shells)
            # 3d. find HB labels
            HBLabels.get_HB_labels(solvent.index, system, HBs, shells)
            if shell.nearest_nonlike_idx is not None:
                # 3e. populate the labels into a dictionary for stats
                # only if a different atom is in the RAD shell
                nearest = system.atoms[shell.nearest_nonlike_idx]
                nearest_resid = nearest.resid
                nearest_resname = nearest.resname
                hb_labels.add_data(
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

    # 4. get the orientational entropy of interfacial waters and save
    #   them to a dictionary
    # TO-DO: add average Nc in Sorient dict
    # Sorient_dict = Orient.get_resid_orientational_entropy_from_dict(
    #     hb_labels.resid_labelled_shell_counts
    # )
    Sorients = Orientations()
    Sorients.add_data(hb_labels)
    Sorient_dict = Sorients.resid_labelled_Sorient
    # 5. Get the vibrational entropy of interfacial waters
    vibrations.add_data(covariances, diagonalise=True)

    return (
        Sorient_dict,
        covariances,
        vibrations,
        frame_solvent_indices,
    )


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


def print_frame_solvent_dicts(frame_solvent_indices: dict):
    """
    Print the interfacial solvent for each analysed frame

    :param frame_solvent_indices: dictionary containing solvent indices in the
        first shell of solute atoms over each frame analysed
    """
    for frame, resname_key in sorted(list(frame_solvent_indices.items())):
        for resname, resid_key in sorted(list(resname_key.items())):
            for resid, solvents in sorted(list(resid_key.items())):
                print(frame, resname, resid, len(solvents), solvents)
