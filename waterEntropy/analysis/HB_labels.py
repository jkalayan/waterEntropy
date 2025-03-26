"""
Store the labelled neighbours that are donated to and accepted from the central
atom in a shell. Shells with the same neighbours are grouped together and
along with the counts for donating to and accepting from each neighbour type.
These labelled HB neighbours are used to calculate orientational
entropy of water molecules.
"""

from waterEntropy.analysis.HB import HBCollection
from waterEntropy.analysis.shells import ShellCollection
from waterEntropy.utils.helpers import nested_dict
import waterEntropy.utils.selections as Selections


class HBLabelCollection:
    """
    Labelled shell counts used for Sorient.
    The counts are placed into two dictionaries used for statistics later,
    the structure of these two dictionaries are as follows:

    dict1 = {"resname": {("labelled_shell"): {"shell_count": 0,
                                "donates_to": {"labelled_donators": 0,},
                                "accepts_from": {"labelled_acceptors": 0,}
                                }}}
    dict2 = {"nearest_resid": {"resname":
                                {("labelled_shell"): {"shell_count": 0,
                                "donates_to": {"labelled_donators": 0,},
                                "accepts_from": {"labelled_acceptors": 0,}
                                }}}
    """

    def __init__(self):
        self.labelled_shell_counts = nested_dict()  # save shell instances in here
        self.resid_labelled_shell_counts = nested_dict()  # save shell instances in here

    def add_data(self, resid, resname, labelled_shell, donates_to, accepts_from):
        # pylint: disable=too-many-arguments
        """Add data to class dictionaries"""
        self.add_shell_counts(resid, resname, labelled_shell)
        self.add_donates_to(resid, resname, labelled_shell, donates_to)
        self.add_accepts_from(resid, resname, labelled_shell, accepts_from)

    def add_shell_counts(self, resid, resname, labelled_shell):
        """
        Add a labelled shell to a dictionary that keeps track of the counts
        for each labelled shell type with constituents alpha-numerically
        ordered

        :param self: class instance
        :param resid: residue id of nearest nonlike atom in the labelled shell
        :param resname: residue name of nearest nonlike atom in the labelled shell
        :param labelled_shell: coordination shell with labelled neighbours
        """
        labelled_shell = tuple(sorted(labelled_shell))
        if "shell_count" not in self.labelled_shell_counts[resname][labelled_shell]:
            self.labelled_shell_counts[resname][labelled_shell]["shell_count"] = 1
        else:
            self.labelled_shell_counts[resname][labelled_shell]["shell_count"] += 1

        if (
            "shell_count"
            not in self.resid_labelled_shell_counts[resid][resname][labelled_shell]
        ):
            self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                "shell_count"
            ] = 1
        else:
            self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                "shell_count"
            ] += 1

    def add_donates_to(self, resid, resname, labelled_shell, donates_to):
        """
        Add a labelled neighbours donated to in a dictionary that keeps
        track of the counts for each labelled shell type with
        constituents alpha-numerically ordered

        :param self: class instance
        :param resid: residue id of nearest nonlike atom in the labelled shell
        :param resname: residue name of nearest nonlike atom in the labelled shell
        :param labelled_shell: coordination shell with labelled neighbours
        :param donates_to: list of labelled neighbours that are donated to
        """
        labelled_shell = tuple(sorted(labelled_shell))
        # donates_to = tuple(sorted(donates_to))
        for a in donates_to:
            if (
                a
                not in self.labelled_shell_counts[resname][labelled_shell]["donates_to"]
            ):
                self.labelled_shell_counts[resname][labelled_shell]["donates_to"][a] = 1
            else:
                self.labelled_shell_counts[resname][labelled_shell]["donates_to"][
                    a
                ] += 1

            if (
                a
                not in self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "donates_to"
                ]
            ):
                self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "donates_to"
                ][a] = 1
            else:
                self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "donates_to"
                ][a] += 1

    def add_accepts_from(self, resid, resname, labelled_shell, accepts_from):
        """
        Add a labelled neighbours accepted from to in a dictionary that keeps
        track of the counts for each labelled shell type with
        constituents alpha-numerically ordered

        :param self: class instance
        :param resid: residue id of nearest nonlike atom in the labelled shell
        :param resname: residue name of nearest nonlike atom in the labelled shell
        :param labelled_shell: coordination shell with labelled neighbours
        :param accepts_from: list of labelled neighbours that are accepted_from
        """
        labelled_shell = tuple(sorted(labelled_shell))
        for d in accepts_from:
            if (
                d
                not in self.labelled_shell_counts[resname][labelled_shell][
                    "accepts_from"
                ]
            ):
                self.labelled_shell_counts[resname][labelled_shell]["accepts_from"][
                    d
                ] = 1
            else:
                self.labelled_shell_counts[resname][labelled_shell]["accepts_from"][
                    d
                ] += 1

            if (
                d
                not in self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "accepts_from"
                ]
            ):
                self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "accepts_from"
                ][d] = 1
            else:
                self.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "accepts_from"
                ][d] += 1


def get_HB_labels(atom_idx: int, system, HBs: HBCollection, shells: ShellCollection):
    """
    For a given central atom, get what UAs it donates and accepts from, then
    find what shell labels these correspond to. Update the shells class
    instance with this information

    :param atom_idx: atom index to find HB labels for
    :param system: mdanalysis instance of all atoms in current frame
    :param HBs: HBCollection class instance
    :param shells: ShellCollection class instance
    """
    shell = shells.find_shell(atom_idx)
    accepts_from_labels = []
    donates_to_labels = []
    if shell:
        # 1.check if HB donating to and accepting from have previously
        # been found
        if shell.labels:
            # 2. Find what UAs the central atom donates to and accepts from
            # with the HBs class instance
            donates_to = HBs.find_acceptor(atom_idx)
            accepts_from = HBs.find_donators(atom_idx)
            if accepts_from:
                for d_idx in accepts_from:
                    # 3. check if UA being accepted from is in shell of
                    # central UA and add label to list
                    bonded_UA = Selections.find_bonded_heavy_atom(d_idx, system)
                    if bonded_UA.index in shell.UA_shell:
                        shell_idx = shell.UA_shell.index(bonded_UA.index)
                        accepts_from_labels.append(shell.labels[shell_idx])
                    else:
                        # don't add donating neighbour if not in central UA
                        # shell
                        continue

            if donates_to:
                # 4. iterate through acceptors and add labels
                for d_idx, a_idx in donates_to.items():
                    acceptor = system.atoms[a_idx]
                    if acceptor.mass < 1.1:
                        acceptor = Selections.find_bonded_heavy_atom(a_idx, system)
                    shell_idx = shell.UA_shell.index(acceptor.index)
                    donates_to_labels.append(shell.labels[shell_idx])
    # 5. Add labelled neighbours to shells instance
    shell.donates_to_labels = donates_to_labels
    shell.accepts_from_labels = accepts_from_labels
