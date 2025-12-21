"""
These functions calculate orientational entropy from labelled
coordination shells
"""

import logging
import time

from dask.distributed import Client

import waterEntropy.analysis.HB as HBond
import waterEntropy.analysis.HB_labels as HBLabels
import waterEntropy.analysis.RAD as RADShell
import waterEntropy.analysis.shell_labels as RADLabels
from waterEntropy.analysis.shells import ShellCollection
from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.entropy.orientations import Orientations
from waterEntropy.entropy.vibrations import Vibrations
import waterEntropy.maths.trig as Trig
from waterEntropy.recipes.forces_torques import get_forces_torques
from waterEntropy.utils.helpers import nested_dict
import waterEntropy.utils.selections as Selections


def find_interfacial_solvent(solutes, system, shells: ShellCollection):
    # pylint: disable=too-many-locals
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
            # 2. DON'T automatically look for RAD shell as this is slow,
            # instead, check if waters are in the distance array before using
            # RAD. Instead, find distances of neighbours first.
            sorted_indices, sorted_distances = Trig.get_sorted_neighbours(
                atom.index, system
            )
            sorted_atoms = Selections.get_selection(
                system, "index", sorted_indices[:20]
            )
            sorted_waters = sorted_atoms.select_atoms("water")
            # 2b. Only get RAD shells if there is a water in the closest X
            # neighbours
            if len(sorted_waters) > 0:
                # 3. find the shell of each UA atom in a molecule
                shell = RADShell.get_RAD_shell(
                    atom, system, shells, sorted_indices, sorted_distances
                )  # get the molecule UA shell
                shell_indices = shell.UA_shell
                # 4. for each neighbour in the RAD shell, find single UA molecules
                shell_atoms = Selections.get_selection(system, "index", shell_indices)
                waters = shell_atoms.select_atoms("water")
                solvent_indices.extend(waters.indices)
    return list(set(solvent_indices))


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


def get_interfacial_shells(system, start: int, end: int, step: int):
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
    frame_solvent_shells = nested_dict()
    # pylint: disable=unused-variable
    for ts in system.trajectory[start:end:step]:
        # print(ts)
        # initialise the RAD and HB class instances to store shell information
        shells = ShellCollection()
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
            # 3a. find RAD shell of interfacial solvent
            shell = RADShell.get_RAD_shell(solvent, system, shells)
            shell.nearest_nonlike_idx = RADLabels.get_nearest_nonlike(shell, system)
            if shell.nearest_nonlike_idx is not None:
                # 3b. populate the shells into a dictionary for stats
                # only if a different atom is in the RAD shell
                nearest = system.atoms[shell.nearest_nonlike_idx]
                frame_solvent_shells = save_solvent_shells(
                    ts.frame,
                    shell.atom_idx,
                    shell.UA_shell,
                    frame_solvent_shells,
                )
    return frame_solvent_shells


def save_solvent_shells(
    frame: int,
    atom_idx: int,
    shell_indices: list,
    frame_solvent_shells: dict,
):
    """
    Save the solvent indices at interfaces per frame into a dictionary

    :param frame: frame number of analysed frame
    :param atom_idx: solvent atom index
    :param nearest_resid: residue of number of nearest solute molecule
    :param nearest_resname: residue name of nearest solute molecule
    :param frame_solvent_indices: the dictionary to populate
    """
    if atom_idx not in frame_solvent_shells[frame]:
        frame_solvent_shells[frame][atom_idx] = shell_indices
    return frame_solvent_shells


def print_frame_solvent_shells(frame_solvent_shells: dict):
    """
    Print the interfacial solvent for each analysed frame

    :param frame_solvent_indices: dictionary containing solvent indices in the
        first shell of solute atoms over each frame analysed
    """
    for frame, atom_idx_key in sorted(list(frame_solvent_shells.items())):
        for atom_idx, shell_indices in sorted(list(atom_idx_key.items())):
            print(frame, atom_idx, shell_indices, len(shell_indices))


def get_interfacial_water_orient_entropy(
    system,
    start: int,
    end: int,
    step: int,
    temperature=298,
    parallel=False,
    client=None,
):
    # pylint: disable=E1121
    # pylint: disable=R0913
    # pylint: disable=too-many-locals
    """
    For a given system, containing the topology and coordinates of molecules,
    find the interfacial water molecules around solutes and calculate their
    orientational entropy if there is a solute atom in the solvent coordination
    shell. This method is a common calling function for the serial and parallel
    implementation and should be the calling point into those functions. Defaults
    to serial.

    :param system: mdanalysis instance of atoms in a frame
    :param start: starting frame number
    :param end: end frame number
    :param step: steps between frames
    :param parallel: set to True to run the parallel calculation
    :param client: dask client instance if custom cluster layout required
    """
    # Initialise data structures to hold processed data.
    frame_solvent_indices = nested_dict()
    covariances = CovarianceCollection()
    hb_labels = HBLabels.HBLabelCollection()
    n_frames = 0
    results = []
    # Steps 1,2 and 3 in function calls.
    indices = list(range(start, end, step))
    if parallel is True:
        # If no Dask client cluster has been setup externally then default to single host.
        if client is None:
            client = Client(
                processes=True, threads_per_worker=1, silence_logs=logging.CRITICAL
            )
            # parallelise over frames by packing them and vars into batches and mapping
            # them across dask workers.
            batch_size = client.scheduler_info()["n_workers"]
            batches = [
                indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
            ]
            for i, batch in enumerate(batches):
                args = [(index, system) for index in batch]
                futures = client.map(_entropy_per_step, args)
                results.extend(client.gather(futures))
                if i < len(batches) - 1:
                    client.restart()
                    time.sleep(2)
        # Otherwise we are on HPC and we can run without the memory fixes.
        else:
            args = [(index, system) for index in indices]
            futures = client.map(_entropy_per_step, args)
            results.extend(client.gather(futures))
        client.close()
    else:
        for index in indices:
            results.append(_entropy_per_step((index, system)))
    # merge the solvent indices dicts
    for res in results:
        frame_solvent_indices.update(res[0])
        covariances.merge(res[1])
        hb_labels.merge(res[2])
        n_frames += res[3]
    # 4. get the orientational entropy of interfacial waters and save
    #   them to a dictionary
    # TO-DO: add average Nc in Sorient dict
    # Sorient_dict = Orient.get_resid_orientational_entropy_from_dict(
    #     hb_labels.resid_labelled_shell_counts
    # )
    # # 5. Get the vibrational entropy of interfacial waters
    # initialise the Vibrations class instance to store vibrational entropies
    Sorients = Orientations()
    Sorients.add_data(hb_labels)
    Sorient_dict = Sorients.resid_labelled_Sorient
    vibrations = Vibrations(temperature)
    vibrations.add_data(covariances, diagonalise=True)

    return (
        Sorient_dict,
        covariances,
        vibrations,
        frame_solvent_indices,
        n_frames,
    )


def _entropy_per_step(args):
    # pylint: disable=too-many-locals
    """
    For a given system, containing the topology and coordinates of molecules,
    find the interfacial water molecules around solutes and calculate their
    orientational entropy if there is a solute atom in the solvent coordination
    shell.

    :param args: tuple of variables containing frame index and MDA system frame.
    """
    # 1. unpack vars
    index, system = args
    ts = system.trajectory[index]
    # 2. redeclare these on the frame by frame basis, and assemble them into the
    # main data structures upon return.
    frame_solvent_indices = nested_dict()
    # 3. initialise the Covariance class instance to store covariance matrices
    covariances = CovarianceCollection()
    hb_labels = HBLabels.HBLabelCollection()
    # and store number of frames analysed
    n_frames = 1
    # 4. initialise the RAD and HB class instances to store shell information
    shells = ShellCollection()
    HBs = HBond.HBCollection()
    # 5. find > 1 UA molecules in system, these are the solutes
    resid_list = Selections.find_solute_molecules(system)
    solutes = Selections.get_selection(system, "resid", resid_list)
    # 6. find the interfacial solvent molecules that are 1 UA in size
    #   and are in the RAD shell of any solute
    solvent_indices = find_interfacial_solvent(solutes, system, shells)
    first_shell_solvent = Selections.get_selection(system, "index", solvent_indices)
    # 7. iterate through first shell solvent and find their RAD shells,
    #   HBing in the shells and shell labels
    for solvent in first_shell_solvent:
        # print(solvent)
        # 8a. find RAD shell of interfacial solvent
        shell = RADShell.get_RAD_shell(solvent, system, shells)
        # 8b. find HBing in the shell
        HBond.get_shell_HBs(shell, system, HBs, shells)
        # 8c. find RAD shell labels
        shell = RADLabels.get_shell_labels(solvent.index, system, shell, shells)
        # 8d. find HB labels
        HBLabels.get_HB_labels(solvent.index, system, HBs, shells)
        if shell.nearest_nonlike_idx is not None:
            # 8e. populate the labels into a dictionary for stats
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
    return frame_solvent_indices, covariances, hb_labels, n_frames
