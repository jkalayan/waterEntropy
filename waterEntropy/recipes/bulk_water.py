"""
These functions calculate orientational entropy from labelled
coordination shells of bulk water
"""

import waterEntropy.analysis.HB as HBond
import waterEntropy.analysis.HB_labels as HBLabels
import waterEntropy.analysis.RAD as RADShell
from waterEntropy.analysis.shells import ShellCollection
from waterEntropy.entropy.convariances import CovarianceCollection
from waterEntropy.entropy.orientations import Orientations
from waterEntropy.entropy.vibrations import Vibrations
from waterEntropy.recipes.forces_torques import get_forces_torques


def get_bulk_water_orient_entropy(
    system, start: int, end: int, step: int, temperature=298
):
    # pylint: disable=too-many-locals
    """
    For a given system, containing the topology and coordinates of molecules,
    find the bulk water and calculate their
    orientational entropy if there is a solute atom in the solvent coordination
    shell.

    :param system: mdanalysis instance of atoms in a frame
    :param start: starting frame number
    :param end: end frame number
    :param step: steps between frames
    """
    # initialise the Covariance class instance to store covariance matrices
    covariances = CovarianceCollection()
    # initialise the Vibrations class instance to store vibrational entropies
    vibrations = Vibrations(temperature)
    hb_labels = HBLabels.HBLabelCollection()
    # pylint: disable=unused-variable
    for ts in system.trajectory[start:end:step]:
        # initialise the RAD and HB class instances to store shell information
        shells = ShellCollection()
        HBs = HBond.HBCollection()
        # select all water molecules in the system
        waters = system.select_atoms("water and mass 2 to 999")
        # 3. iterate through first shell solvent and find their RAD shells,
        #   HBing in the shells and shell labels
        for solvent in waters:
            # 3a. find RAD shell of interfacial solvent
            shell = RADShell.get_RAD_shell(solvent, system, shells)
            # 3b. Set RAD shell labels as resname of neighbour
            shell.labels = [system.atoms[n].resname for n in shell.UA_shell]
            # check shell only contains only solvent molecules
            if set(shell.labels) == {solvent.resname}:
                # 3c. find HBing in the shell
                HBond.get_shell_HBs(shell, system, HBs, shells)
                # 3d. find HB labels
                HBLabels.get_HB_labels(solvent.index, system, HBs, shells)
                hb_labels.add_data(
                    f"{solvent.resname}",
                    f"{solvent.resname}",
                    shell.labels,
                    shell.donates_to_labels,
                    shell.accepts_from_labels,
                )
                # 3e. calculate the running average of force and torque
                # covariance matrices
                solvent_molecule = system.atoms[solvent.index].fragment  # get molecule
                get_forces_torques(
                    covariances,
                    solvent_molecule,
                    f"{solvent.resname}",
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
    )
