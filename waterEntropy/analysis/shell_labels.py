"""
Label neighbours in a coordination shell based on what they are and what their
neighbours are.
"""

from waterEntropy.analysis.HB import HBCollection
import waterEntropy.analysis.RAD as RADShell
from waterEntropy.analysis.shells import ShellCollection
import waterEntropy.utils.selections as Selections


def get_shell_labels(
    atom_idx: int, system, shell, shells: ShellCollection, HBs: HBCollection = None
):
    # pylint: disable=too-many-locals
    # pylint: disable=unused-argument
    # pylint: disable=too-many-branches
    """
    Get the shell labels of an atoms shell based on the following:
    For a central UA, rank its coordination shell by proximity to that
    central UA's nearest non-like molecule UA.

    * '#_RESNAME' = RAD shell from same molecule type, when nearest nonlike resid is the same as the reference.

    * 'X_RESNAME' = when same molecule type has different nearest nonlike resid.

    * 'RESNAME' = when molecule of different type is in RAD shell.

    * '0_RESNAME' = closest different type molecule in RAD shell. (the one its assigned to, its nearest non-like!)

    :param atom_idx: atom index of central atom in coordination shell
    :param system: mdanalysis instance of atoms in a frame
    :param shell: shell instance of atom_idx
    :param shells: ShellCollection instance
    """
    center = system.atoms[atom_idx]
    # 1. find the closest different UA in a shell
    #   different = not the same resname
    nearest_nonlike_idx = get_nearest_nonlike(shell, system)

    # find the list of nonlikes in the the UA shell
    nonlikes = get_nonlike_list(shell, system)

    # # 1b. OPTIONAL: override the nearest_nonlike if strongest HB is with a solute
    # strongest_HB = get_strongest_HB(atom_idx, system, HBs)[0]
    # if strongest_HB is not None:
    #     if strongest_HB.resname != center.resname:
    #         nearest_nonlike_idx = strongest_HB.index
    #     else:
    #         nearest_nonlike_idx = None
    #         shell.nearest_nonlike_idx = None

    # 2. only find labels if a solute is in the shell
    if nearest_nonlike_idx is not None:
        nearest_nonlike = system.atoms[nearest_nonlike_idx]
        shell_labels = []
        N_w = 0
        for n in shell.UA_shell:
            neighbour = system.atoms[n]
            # 3a. label nearest nonlike atom as "0_RESNAME"
            if neighbour.index == nearest_nonlike.index:
                # shell_labels.append(f"0_{neighbour.resname}")

                # only allow single UA molecules to be distinguishable
                # in shell
                if len(neighbour.fragment) == 1:
                    shell_labels.append(f"0_{neighbour.resname}")
                else:
                    shell_labels.append("SOL")

            # 3b. label other nonlike atoms as "RESNAME"
            if (
                neighbour.index != nearest_nonlike.index
                and neighbour.resname != center.resname
            ):
                # shell_labels.append(neighbour.resname)

                # only allow single UA molecules to be distinguishable
                # in shell
                if len(neighbour.fragment) == 1:
                    shell_labels.append(neighbour.resname)
                else:
                    shell_labels.append("SOL")

            # 3c. find RAD shells for shell constituents with same resname
            # as central atom
            if (
                neighbour.index != nearest_nonlike.index
                and neighbour.resname == center.resname
            ):
                N_w += 1
                neighbour_shell = shells.find_shell(neighbour.index)
                if not neighbour_shell:
                    neighbour_shell = RADShell.get_RAD_shell(neighbour, system, shells)
                # 3d. find nearest nonlike of neighbours with same resname
                # as central atom
                neighbour_nearest_nonlike_idx = get_nearest_nonlike(
                    neighbour_shell, system
                )
                # 3e. if neighbour has a pure shell, then it is in the second
                # shell of the nearest nonlike
                if neighbour_nearest_nonlike_idx is None:
                    shell_labels.append(f"2_{neighbour.resname}")
                else:
                    # 3f. if neighbours nearest nonlike is the same atom as
                    # central atom, assume it is in the first shell
                    # if neighbour_nearest_nonlike_idx == nearest_nonlike_idx:
                    neighbour_nearest_nonlike = system.atoms[
                        neighbour_nearest_nonlike_idx
                    ]
                    # 3g. if neighbours nearest nonlike is in the same resid as
                    # central atom, assume it is in the first shell
                    if neighbour_nearest_nonlike.resid == nearest_nonlike.resid:
                        shell_labels.append(f"1_{neighbour.resname}")
                    else:
                        # 3h. if neighbours nearest nonlike is not the same resid
                        # as central nearest resid,  it is in the first shell
                        # of a different resid and labelled as "X_RESNAME"
                        shell_labels.append(f"X_{neighbour.resname}")
        shell.labels = shell_labels  # sorted(shell_labels) #don't sort yet
        shell.nearest_nonlike_idx = nearest_nonlike.index
        shell.nonlikes_idxs = nonlikes
        shell.N_w = N_w
    return shell


def get_nearest_nonlike(shell, system):
    """
    For a given shell, find the closest neighbour that is not the same
    atom/molecule type as the central united atom.

    :param shell: shell instance of an atom
    :param system: mdanalysis instance of atoms in a frame
    """
    nearest_nonlike_idx = None
    center = system.atoms[shell.atom_idx]
    for n in shell.UA_shell:
        neighbour = system.atoms[n]
        if neighbour.resname != center.resname and neighbour.type != center.type:
            nearest_nonlike_idx = n
            break
    return nearest_nonlike_idx


def get_nonlike_list(shell, system):
    """
    For a given shell, find neighbours that are not the same
    atom/molecule type as the central united atom, save indices of
    these non-like neighbours in a list.

    :param shell: shell instance of an atom
    :param system: mdanalysis instance of atoms in a frame
    """
    nonlikes = []
    center = system.atoms[shell.atom_idx]
    for n in shell.UA_shell:
        neighbour = system.atoms[n]
        if neighbour.resname != center.resname and neighbour.type != center.type:
            nonlikes.append(n)
    return nonlikes


def get_strongest_HB(atom_idx: int, system, HBs: HBCollection):
    """
    For a given water, find its strongest HB to a neighbour, could be accepting
    from or donating to.

    :param shell: shell instance of an atom
    :param system: mdanalysis instance of atoms in a frame
    """

    strongest_HB = [None, 0]
    # 1. Find what UAs the central atom donates to and accepts from
    # with the HBs class instance
    donates_to = HBs.find_acceptor(atom_idx)
    accepts_from = HBs.find_donators(atom_idx)
    if accepts_from:
        for d_idx, HB_strength in accepts_from:
            # 2. Find heavy atom bonded to donating H atom
            bonded_UA = Selections.find_bonded_heavy_atom(d_idx, system)
            if HB_strength < strongest_HB[1]:
                strongest_HB = [bonded_UA, HB_strength]

    if donates_to:
        # 3. iterate through acceptors
        for d_idx, (a_idx, HB_strength) in donates_to.items():
            acceptor = system.atoms[a_idx]
            if acceptor.mass < 1.1:
                acceptor = Selections.find_bonded_heavy_atom(a_idx, system)
            if HB_strength < strongest_HB[1]:
                strongest_HB = [acceptor, HB_strength]
    # 4. Return strongest HB atom and strength
    return strongest_HB
