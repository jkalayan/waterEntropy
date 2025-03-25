"""
Store the labelled neighbours that are donated to and accepted from the central
atom in a shell. These labelled HB neighbours are used to calculate orientational
entropy of water molecules.
"""

from waterEntropy.utils.helpers import nested_dict


class HBLabels:
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
        # accepts_from = tuple(sorted(accepts_from))
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
