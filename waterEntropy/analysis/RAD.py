"""
These functions calculate coordination shells using RAD the relative
angular distance.
"""

import numpy as np

import waterEntropy.maths.trig as Trig
import waterEntropy.utils.selections as Selections


class Shell:
    """Represent a single atom shell with dynamic properties, which are set in
    the ShellCollection class."""

    def __init__(self, atom_idx):
        object.__setattr__(self, "atom_idx", atom_idx)  # Store atom index directly
        object.__setattr__(self, "properties", {})  # Internal dictionary

    def __getattr__(self, key: str):
        """Allow dot notation access to stored properties.

        :param key: name of the property in the properties dictionary
        """
        if key in self.properties:
            return self.properties[key]
        raise AttributeError(f"Property '{key}' not found for atom {self.atom_idx}")

    def __setattr__(self, key: str, value):
        """Allow adding properties dynamically, except for 'atom_idx'.

        :param key: name of the property being added to the properties dictionary
        :param value: value for the key being added to the properties dictionary
        """
        if key == "atom_idx":  # Ensure atom index remains immutable
            object.__setattr__(self, key, value)
        else:
            self.properties[key] = value  # Store additional properties

    def __repr__(self):
        """Print the atom and its properties."""
        return f"Shell(atom_idx={self.atom_idx}, properties={self.properties})"


class ShellCollection:
    """Manage multiple atom shells and allow adding/updating properties."""

    def __init__(self):
        self.shells = {}  # Dictionary storing Atom objects indexed by atom index

    def add_data(self, atom_idx: int, UA_shell: list[int]):
        """Add a new atom shell to the collection if it doesn't exist and set
        various properties used to describe the class.

        :param atom_idx: the heavy atom index of the central atom in a shell
        :param UA_shell: the list of heavy atom indices in the shell of atom_idx
        """
        if atom_idx not in self.shells:
            self.shells[atom_idx] = Shell(atom_idx)
            self.set_property(atom_idx, "atom_idx", atom_idx)
            self.set_property(atom_idx, "UA_shell", UA_shell)
            self.set_property(atom_idx, "nearest_nonlike_idx", None)
            self.set_property(atom_idx, "labels", None)
            self.set_property(atom_idx, "donates_to_labels", None)
            self.set_property(atom_idx, "accepts_from_labels", None)

    def set_property(self, atom_idx, key: str, value):
        """Set a property for a specific atom shell, ensuring it exists first.

        :param atom_idx: the heavy atom index of the central atom in a shell
        :param key: the name of the key being added to the shells dictionary
        :param value: the value of the key being added to the shells dictionary
        """
        if atom_idx not in self.shells:
            self.shells[atom_idx] = Shell(atom_idx)  # Auto-create atom if missing
        setattr(self.shells[atom_idx], key, value)

    def find_shell(self, atom_idx: int):
        """
        Get the shell instance if atom_idx is a key in shells dictionary

        :param cls: instance of RAD class
        :param atom_idx: index of central atom in shell
        """
        return self.shells.get(atom_idx, None)

    def __repr__(self):
        """Return a dictionary-like representation of stored atoms."""
        return repr(self.shells)


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
            shell = get_RAD_shell(atom, system, shells)  # get the molecule UA shell
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


def get_RAD_shell(UA, system, shells: ShellCollection):
    """
    For a given united atom, find its RAD shell, returning the atom indices
    for the heavy atoms that are in its shell.

    :param UA: mdanalysis instance of a united atom in a frame
    :param system: mdanalysis instance of atoms in a frame
    :param shells: ShellCollection instance
    """
    # 1. first check if a shell has already been found for this UA
    shell = shells.find_shell(UA.index)
    if not shell:
        # 2. get the nearest neighbours for the UA, sorted from closest to
        # furthest
        sorted_indices, sorted_distances = get_sorted_neighbours(UA.index, system)
        # 3. now find the RAD shell of the UA
        shell_indices = get_RAD_neighbours(
            UA.position, sorted_indices, sorted_distances, system
        )
        # 4. populate the class instance for RAD shells
        shells.add_data(UA.index, shell_indices)
        shell = shells.find_shell(UA.index)
    return shell


def get_sorted_neighbours(i_idx: int, system):
    """
    For a given atom, find the united atoms in its coordination shell. The
    shell instance from the RAD class is created for the given atom

    :param i_idx: idx of atom i
    :param system: mdanalysis instance of atoms in a frame
    """
    i_coords = system.atoms.positions[i_idx]
    # 1. get the heavy atom neighbour distances within a given distance cutoff
    # CHECK Find out which of the options below is better for RAD shells
    #       Should the central atom bonded UAs be allowed to block?
    #       This was not done in original code, keep the same here
    neighbours = system.select_atoms(
        f"""mass 2 to 999 and not index {i_idx}
                                    and not bonded index {i_idx}"""
        # f"""mass 2 to 999 and not index {i_idx}"""  # bonded UAs can block
    )
    sorted_indices, sorted_distances = Trig.get_neighbourlist(
        i_coords, neighbours.atoms, system.dimensions, max_cutoff=25
    )
    return sorted_indices, sorted_distances


def get_RAD_neighbours(i_coords, sorted_indices, sorted_distances, system):
    # pylint: disable=too-many-locals
    r"""
    For a given set of atom coordinates, find its RAD shell from the distance
    sorted atom list, truncated to the closests 30 atoms.

    This function calculates coordination shells using RAD the relative
    angular distance, as defined first in DOI:10.1063/1.4961439
    where united atoms (heavy atom + bonded Hydrogens) are defined as neighbours if
    they fulfil the following condition:

    .. math::
        \Bigg(\frac{1}{r_{ij}}\Bigg)^2>\Bigg(\frac{1}{r_{ik}}\Bigg)^2 \\cos \theta_{jik}

    For a given particle :math:`i`, neighbour :math:`j` is in its coordination
    shell if :math:`k` is not blocking particle :math:`j`. In this implementation
    of RAD, we enforce symmetry, whereby neighbouring particles must be in each
    others coordination shells.

    :param i_coords: xyz coordinates of atom i
    :param sorted_indices: list of atom indices sorted from closest to
        furthest from atom i
    :param sorted_distances: list of atom distances sorted from closest to
        furthest from atom i
    :param system: mdanalysis instance of atoms in a frame
    """
    # 1. truncate neighbour list to closest 30 united atoms
    range_limit = min(len(sorted_distances), 30)
    shell = []
    count = -1
    # 2. iterate through neighbours from closest to furthest
    for y in sorted_indices[:range_limit]:
        count += 1
        y_idx = np.where(sorted_indices == y)[0][0]
        j = system.atoms.indices[y]
        j_coords = system.atoms.positions[y]
        rij = sorted_distances[y_idx]
        blocked = False
        # 3. iterate through neighbours than atom j and check if they block
        # it from atom i
        for z in sorted_indices[:count]:  # only closer UAs can block
            z_idx = np.where(sorted_indices == z)[0][0]
            k_coords = system.atoms.positions[z]
            rik = sorted_distances[z_idx]
            # 4. find the angle jik
            costheta_jik = Trig.get_angle(
                j_coords, i_coords, k_coords, system.dimensions[:3]
            )
            if np.isnan(costheta_jik):
                break
            # 5. check if k blocks j from i
            LHS = (1 / rij) ** 2
            RHS = ((1 / rik) ** 2) * costheta_jik
            if LHS < RHS:
                blocked = True
                break
        # 6. if j is not blocked from i by k, then its in i's shell
        if blocked is False:
            shell.append(j)
    return shell


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
                    neighbour_shell = get_RAD_shell(neighbour, system, shells)
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
