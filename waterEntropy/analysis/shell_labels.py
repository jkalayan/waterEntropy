"""
Label neighbours in a coordination shell based on what they are and what their
neighbours are.
"""

import waterEntropy.analysis.RAD as RADShell
from waterEntropy.analysis.shells import ShellCollection


def get_shell_labels(atom_idx: int, system, shell, shells: ShellCollection):
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
    # 2. only find labels if a solute is in the shell
    if nearest_nonlike_idx is not None:
        nearest_nonlike = system.atoms[nearest_nonlike_idx]
        shell_labels = []
        for n in shell.UA_shell:
            neighbour = system.atoms[n]
            # 3a. label nearest nonlike atom as "0_RESNAME"
            if neighbour.index == nearest_nonlike.index:
                shell_labels.append(f"0_{neighbour.resname}")
            # 3b. label other nonlike atoms as "RESNAME"
            if (
                neighbour.index != nearest_nonlike.index
                and neighbour.resname != center.resname
            ):
                shell_labels.append(neighbour.resname)
            # 3c. find RAD shells for shell constituents with same resname
            # as central atom
            if (
                neighbour.index != nearest_nonlike.index
                and neighbour.resname == center.resname
            ):
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
