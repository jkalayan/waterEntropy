"""
These functions calculate hydrogen bonding within a shell.
"""

import numpy as np

import waterEntropy.analysis.RAD as RADShell
import waterEntropy.maths.trig as Trig
from waterEntropy.utils.helpers import nested_dict
import waterEntropy.utils.selections as Selections


class HBCollection:
    """
    Class for hydrogen bond donors and their acceptors
    """

    def __init__(self):
        self.donating_to = (
            nested_dict()
        )  # structure: UA_idx[donating_idx] = accepting_idx
        self.accepting_from = (
            nested_dict()
        )  # structure: UA_idx[accepting_idx] = [donating_idx1,]

    def add_data(self, UA_idx: int, donator_idx: int, acceptor_idx: int):
        """
        For a given donator index, save its acceptor index in the donating_to
        dictionary. And for that acceptor, save the donator index as a list in
        the accepting from dictionary.

        :param self: class instance
        :param UA_idx: atom index of the UA of the bonded donating hydrogen
        :param donator_idx: atom index of the donator bonded to the UA
        :param acceptor_idx: atom index of the UA accepting the
            hydrogen bond in the coordination shell
        """
        if donator_idx not in self.donating_to[UA_idx]:
            self.donating_to[UA_idx][donator_idx] = acceptor_idx
        if acceptor_idx not in self.accepting_from:
            self.accepting_from[acceptor_idx] = []
        if donator_idx not in self.accepting_from[acceptor_idx]:
            self.accepting_from[acceptor_idx].append(donator_idx)

    def find_donators(self, UA_idx: int):
        """
        Find the donators for a given residue, this returns a dict where the
        key is the accepting atom index and the value is a list of the atom
        indices that donate to it.

        :param cls: class instance
        :param UA_idx: atom index of UA being donated to
        """
        return self.accepting_from.get(UA_idx, None)

    def find_acceptor(self, UA_idx: int):
        """
        Find the acceptors for a given resid, this returns a dictionary where
        the key is the donator index and the value is the acceptor index

        :param cls: class instance
        :param UA_idx: atom index of UA accepting HBs
        """
        return self.donating_to.get(UA_idx, None)


def get_shell_HBs(shell, system, HBs: HBCollection, shells: RADShell.ShellCollection):
    """
    For a given UA and its coordination shell neighbours, find what the central
    UA donates to and accepts from in its shell.

    :param shell: the instance for class waterEntropy.neighbours.RAD.RAD
        containing coordination shell neighbours
    :param system: mdanalysis instance of all atoms in current frame
    :param HBs: HBCollection instance
    :param shells: ShellCollection instance
    """
    # 1. first check that HB donations haven't already been found
    donates_to = HBs.find_acceptor(shell.atom_idx)
    if not donates_to:
        get_shell_HB_acceptors(shell, system, HBs)
        donates_to = HBs.find_acceptor(shell.atom_idx)
    # 2. now iterate through shell and find shells of shell neighbours
    for n_idx in shell.UA_shell:
        neighbour_shell = shells.find_shell(n_idx)
        if not neighbour_shell:
            neighbour_shell = RADShell.get_RAD_shell(
                system.atoms[n_idx], system, shells
            )
        # 3. find what each shell neighbour donates to in their shell
        neighbour_donates_to = HBs.find_acceptor(n_idx)
        if not neighbour_donates_to:
            get_shell_HB_acceptors(neighbour_shell, system, HBs)
            neighbour_donates_to = HBs.find_acceptor(n_idx)


def get_HB_labels(
    atom_idx: int, system, HBs: HBCollection, shells: RADShell.ShellCollection
):
    """
    For a given central atom, get what UA it donates and accepts from, then
    find what shell labels these correspond to.

    :param atom_idx: atom index to find HB labels for
    :param system: mdanalysis instance of all atoms in current frame
    :param HBs: HBCollection instance
    :param shells: ShellCollection instance
    """
    shell = shells.find_shell(atom_idx)
    accepts_from_labels = []
    donates_to_labels = []
    if shell:
        if shell.labels:
            # check if HB donating to and accepting from have previously
            # been found
            donates_to = HBs.find_acceptor(atom_idx)
            accepts_from = HBs.find_donators(atom_idx)
            if accepts_from:
                for d_idx in accepts_from:
                    # check if UA being accepted from is in shell of
                    # central UA
                    bonded_UA = Selections.find_bonded_heavy_atom(d_idx, system)
                    if bonded_UA.index in shell.UA_shell:
                        shell_idx = shell.UA_shell.index(bonded_UA.index)
                        accepts_from_labels.append(shell.labels[shell_idx])
                    else:
                        # don't add donating neighbour if not in central UA
                        # shell
                        continue

            if donates_to:
                # iterate through acceptors and add labels
                for d_idx, a_idx in donates_to.items():
                    acceptor = system.atoms[a_idx]
                    if acceptor.mass < 1.1:
                        acceptor = Selections.find_bonded_heavy_atom(a_idx, system)
                    shell_idx = shell.UA_shell.index(acceptor.index)
                    donates_to_labels.append(shell.labels[shell_idx])
    shell.donates_to_labels = donates_to_labels
    shell.accepts_from_labels = accepts_from_labels


def get_shell_HB_acceptors(shell, system, HBs: HBCollection):
    # pylint: disable=too-many-locals
    """
    Find the hydrogen bond acceptors for the central UA hydrogens that are
    electropositive. The assumption is made that hydrogen bonding can only
    occur inside the coordination shell, this is needed for shell neighbour
    labelling used in orientational entropy calculations

    :param shell: the instance for containing coordination shell neighbours
    :param system: mdanalysis instance of all atoms in current frame
    :param HBs: HBCollection instance
    """
    X_idx = shell.atom_idx  # shell central heavy atom
    heavy_atom = system.atoms[X_idx]
    # 1. iterate over atoms bonded to atom X
    for i in heavy_atom.bonds:
        D_idx = i.indices[1]
        bonded = system.atoms[D_idx]
        # 2. find hydrogen atoms bonded to heavy atom with positive charge
        if bonded.mass < 1.1 and bonded.mass > 1.0 and bonded.charge > 0:
            donator = system.atoms[D_idx]
            # 3. set starting hydrogen bond term and possible acceptor to a large number
            # the acceptor that gives the lowest relative charge is the acceptor
            current_relative_charge = 99
            current_acceptor = False
            # 4. get shell neighbours ordered by ascending distance
            sorted_indices, sorted_distances = get_shell_neighbour_selection(
                shell, donator, system
            )
            for A_idx, DA_distance in zip(sorted_indices, sorted_distances):
                # 5. check if neighbouring atom in shell is an acceptor,
                # if so override current possible acceptor
                acceptor = system.atoms[A_idx]
                if not current_acceptor:
                    current_acceptor = acceptor
                relative_charge, XDA_angle = get_HB_terms(
                    heavy_atom, donator, acceptor, DA_distance, system.dimensions[:3]
                )
                # 6. Check if an atom is a possible hydrogen bond acceptor
                if relative_charge < current_relative_charge and float(XDA_angle) > 90:
                    current_relative_charge = relative_charge
                    current_acceptor = acceptor
            # 7. create a new object for hydrogen bonding in a RAD shell
            HBs.add_data(X_idx, D_idx, current_acceptor.index)


def get_shell_neighbour_selection(shell, donator, system, heavy_atoms=True):
    """
    get shell neighbours ordered by ascending distance, this is used for
    finding possible hydrogen bonding neighbours

    :param shell: the instance for class waterEntropy.neighbours.RAD.RAD
        containing coordination shell neighbours
    :param donator: the mdanalysis object for the donator
    :param system: mdanalysis instance of all atoms in current frame
    :param heavy_atoms: consider heavy atoms in a shell as neighbours
    """
    # Can only donate to heavy atoms in the shell
    neighbours = Selections.get_selection(system, "index", shell.UA_shell)
    if not heavy_atoms:
        # get all atoms in the shell, included bonded to
        all_shell_bonded = neighbours[:].bonds.indices
        all_shell_indices = list(set().union(*all_shell_bonded))
        # can donate to any atom in the shell
        neighbours = Selections.get_selection(system, "index", all_shell_indices)
    # can donate to any neighbours outside a shell, not used
    # neighbours = system.select_atoms(f"all and not index {donator.index} and not bonded index {donator.index}")
    sorted_indices, sorted_distances = Trig.get_neighbourlist(
        donator.position, neighbours, system.dimensions, max_cutoff=10
    )
    return sorted_indices, sorted_distances


def get_HB_terms(heavy_atom, donator, acceptor, DA_distance, dimensions):
    """
    Get two terms for calculating the hydrogen bond between a donator and
    potential acceptor in a coordination shell.
    For a hydrogen bond to form, the following criteria need to be met:

    1. Angle between heavy atom bonded to donator (X), the donator (D) and
    the acceptor (A) is over 90 degrees

    2. The relative charge is most negative over all other neighbours in a
    coordination shell

    :param heavy_atom: the mdanalysis instance for the heavy atom bonded to
        the donator
    :param donator: the mdanalysis instance for the donator
    :param acceptor: the mdanalysis instance for the possible acceptor
    :param DA_distance: the distance between donor and acceptor
    :param dimensions: the dimensions of the simulation box
    """
    relative_charge = (float(donator.charge) * float(acceptor.charge)) / float(
        DA_distance**2
    )
    cosine_angle = Trig.get_angle(
        heavy_atom.position, donator.position, acceptor.position, dimensions
    )
    XDA_angle = np.degrees(np.arccos(cosine_angle))

    return relative_charge, XDA_angle
