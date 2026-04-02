"""
Store orientational entropies here
"""

from collections import Counter
import textwrap

import numpy as np

import waterEntropy.analysis.HB_labels as HBLabels
from waterEntropy.utils.helpers import nested_dict


class WaterOrientCalculator:
    """Calculate orientational entropy of water molecules here"""

    GAS_CONSTANT = 8.314  # (J/ (K mol)

    def __init__(self):
        self.Sorient = 0
        self.Nc_eff = 0
        self.pbias_ave = 0

    def add_data(self, shell_label: list, shell_values: dict):
        """
        Calculate the orientational entropy for water molecules with a
        specific shell type/environment.

        :param shell_label: the list or tuple of the shell labels of a
            coordination shell
        :param shell_values: the dictionary containing information about what
            neighbour types are donated to and accepted from
        """
        degeneracy = self.get_shell_degeneracy(shell_label)
        pD_dict = self.get_donor_acceptor_probabilities(shell_values, "donates_to")
        pA_dict = self.get_donor_acceptor_probabilities(shell_values, "accepts_from")
        Nc_eff, pbias_ave = self.get_reduced_neighbours_biases(
            degeneracy, pD_dict, pA_dict
        )
        self.Sorient = self.get_orientation_S(Nc_eff, pbias_ave)
        self.Nc_eff = Nc_eff
        self.pbias_ave = pbias_ave

    def get_non_HB_data(self, Nc):
        """
        Calculate water orientational entropies without accounting for
        HB biasing, using the following equation (as in the host-guest systems
        paper, https://doi.org/10.1007/s10822-021-00406-5):

        .. math::
            S_\\mathrm{orient} = \\ln \\Bigg(N_c ^ {(3 / 2)} \times \\pi ^ {0.5}
            \times \frac{p_\\mathrm{corr}}{\\sigma} \\Bigg)

        where :math:`N_{c}` is the number of UAs in the coordination shell,
        :math:`p_\\mathrm{corr}` is the probability that the neighboring
        molecules are oriented suitably for each solute, which is set to 0.25
        for water. And :math:`\\sigma` is the symmetry number for water which
        is 2.

        """
        S_orient = 0
        if Nc != 0:
            S_orient = np.log((Nc) ** (3 / 2) * np.pi**0.5 * 0.25 / 2)
        S_orient = max(S_orient, 0)
        return S_orient * self.GAS_CONSTANT

    def get_shell_degeneracy(self, shell_label: list):
        """
        For a given labelled shell, find the degeneracy, i.e. count of each
        unique label in the shell

        :param shell_label: the list or tuple of the shell labels of a
            coordination shell
        """
        return Counter(shell_label)

    def get_donor_acceptor_probabilities(self, HB_vals_dict: dict, hb_selection: str):
        """
        For a given labelled shell type, find the probability of accepting from
        or donating to a given neighbour type over all other donors or acceptors

        :param vals_dict: dictionary containing the HB donors or acceptors and
            how often they occur for a given neighbour in a shell e.g.
            {"donates_to": {"labelled_donators": 0,},
            "accepts_from": {"labelled_acceptors": 0,}}
        :param degeneracy: dictionary containing the counts for each neighbour
            type in a shell
        :param hb_selection: string for determining which HB type to analyse,
            options are either "donates_to" or "accepts_from"
        """
        pHB_dict = nested_dict()
        if hb_selection in HB_vals_dict:
            total_hb_count = 0
            for c in HB_vals_dict[hb_selection].values():
                total_hb_count += c
            for hb_neighbour, count in sorted(list(HB_vals_dict[hb_selection].items())):
                pHB = self.get_hb_probability(count, total_hb_count)
                pHB_dict[hb_neighbour] = [pHB, count]
        return pHB_dict

    def get_reduced_neighbours_biases(
        self, degeneracy: dict, pD_dict: dict, pA_dict: dict
    ):
        # pylint: disable=too-many-locals
        r"""
        For a given labelled shell and the dictionary containing the counts for
        each neighbour type in the shell, the dictionaries for the acceptor and
        donator probabilities, use these to find the probability of accepting from
        or donating to over all other HBing to that particular neighbour type i:

        .. math::
            ppD_i = \frac{pD_i}{(pD_i + pA_i)}

        :param degeneracy: dictionary of shell constituent and count
        :param pD_dict: dictionary of neighbours donated to and how often that occurs
        :param pA_dict: dictionary of neighbours accepted from and how often that
            occurs
        """
        Nc_eff = 0
        pbiases = []
        # 1. iterate through each neighbour and the counts for how often they occur
        # in a shell
        for i, N_i in degeneracy.items():
            # 2. find if neighbour i is donated to or accepted from
            d = pD_dict[i] or [0, 0]
            a = pA_dict[i] or [0, 0]
            # 3. the donor/acceptor probabilities for a given neighbour i
            pD_i = d[0]
            pA_i = a[0]
            # 4. if the neighbour type i has both been donated to and accepted from
            # then work out probabilities of accepting from vs donating to
            if pD_i != 0 and pA_i != 0:
                sum_p = pD_i + pA_i
                # 5. work out the probabilities to donate to and accept from
                # over the sum of probabilities to donate and accept
                ppD_i = pD_i / sum_p
                ppA_i = pA_i / sum_p
                # 6. work out the effective number of available neighbour type i
                # where 0.25 is no bias in donating to/accepting from a given
                # neighbour
                N_i_eff = ppD_i * ppA_i / 0.25 * N_i
                # 7. sum the effective neighbours for all neighbour types i
                Nc_eff += N_i_eff
                # 8. work out the bias of donating vs accepting from neighbours of
                # type i, if both are equally likely, pbias = 0.25, if one is
                # more likely than the other the pbais < 0.25
                pbias = ppD_i * ppA_i
                # 9. append pbiases for each occurance of neighbour type i to a
                # list to get the averages later
                # for x in range(0, N_i):
                #     pbiases.append(pbias)
                pbiases.extend([pbias] * N_i)
            else:
                # 10. if a neighbour type i is not both donated to AND accepted from,
                # then it is not a neighbour involved in orientational
                # motion of the central UA and instead is either an anchor
                # or not involved in hydrogen bonding.
                # for x in range(0, N_i):
                #     pbiases.append(0)
                pbiases.extend([0] * N_i)

        # 11. find the average HB bias over all neighbours in the shell, where
        # zeros count towards the average
        pbias_ave = 0
        if len(pbiases) != 0:
            pbias_ave = sum(pbiases) / len(pbiases)

        return Nc_eff, pbias_ave

    def get_orientation_S(self, Nc_eff: float, pbias_ave: float):
        r"""
        Get the orientational entropy of water molecules, or any single UA molecule
        containing two hydrogen bond donors.

        .. math::
            S_\mathrm{orient} = \ln \Bigg(N_{c_\mathrm{eff}} ^ {(3 / 2)} \times \pi ^ {0.5}
            \times \frac{p_\mathrm{bias\_ave}}{2} \Bigg)

        This equation is modified from the previous theory of water molecule
        orientational entropy, where hydrogen bonding is accounted for by reducing
        the number of available neighbours.
        Here, the coordination shell neighbours available to hydrogen bond with
        are reduced to :math:`N_{c_\mathrm{eff}}`. The reduction in available
        HBing neighbours is
        calculated from statistics gathered from simulation trajectories, where
        neighbour types that are donated to or accepted from are counted.

        .. math::
            N_{c_\mathrm{eff}} = \sum_i \bigg((ppD_i \times ppA_i) \times \frac{N_i}{0.25} \bigg)

        where :math:`ppD_i` is the probability to donate to a given neighbour
        type :math:`i` compared to accepting from the same neighbour type.
        :math:`ppA_i` is the equivalent probability for accepting from neighbour
        type :math:`i`.

        :math:`p_\mathrm{bias\_ave}` is the average bias in accepting from and donating to
        a given neighbour type :math:`i`. If it is equally likely to donate to and
        accept from all neighbour types, then :math:`p_\mathrm{bias\_ave}=0.25`, but if there is
        any bias in preferentially donating to, accepting from or not HBing to a
        neighbour at all, then :math:`p_\mathrm{bias\_ave} < 0.25`.

        .. math::
            p_\mathrm{bias\_ave} = \frac{\sum_i(ppD_i * ppA_i)}{N_c}

        Both :math:`N_{c_\mathrm{eff}}` and :math:`p_\mathrm{bias\_ave}` are
        used to reduce the orientational entropy of the central molecule by
        accounting for the HBing observed in a simulation.

        :param Nc_eff: effective number of available neighbours to rotate around
        :param pbias_ave: the average biasing in hydrogen bonding donating or
            accepting over all shell neighbours
        """
        S_orient = 0
        if Nc_eff != 0:
            S_orient = np.log((Nc_eff) ** (3 / 2) * np.pi**0.5 * pbias_ave / 2)
        S_orient = max(S_orient, 0)
        return S_orient * self.GAS_CONSTANT

    def get_hb_probability(self, count: int, total_count: int):
        r"""
        For a given HB donor/acceptor, find the probablity of that HB occurring
        in a given unique labelled shell over all the other acceptors or donors,
        so

        .. math::
            pA_i = \frac{N_{i_A}}{\sum_i(N_{i_A})}

        or

        .. math::
            pD_i = \frac{N_{i_D}}{\sum_i(N_{i_D})}

        :param count: number of HB donors/acceptors for a given neighbour type
        :param total_count: total number of either acceptor or donor counts over
            all neighbour types
        """
        p_i = count / total_count
        return p_i


class Orientations:
    """
    Store orientational entropies here
    """

    def __init__(self):
        self.resid_labelled_Sorient = nested_dict()
        self.resname_labelled_Sorient = nested_dict()

    def add_data(self, hb_labels: HBLabels):
        """Add orientational entropy data"""
        self.get_orientational_entropy_from_dict(
            hb_labels.labelled_shell_counts, self.resname_labelled_Sorient
        )
        self.get_resid_orientational_entropy_from_dict(
            hb_labels.resid_labelled_shell_counts
        )

    def get_resid_orientational_entropy_from_dict(self, resid_labelled_dict: dict):
        """
        For a given dictionary containing labelled shells and HBing within the
        shell with format:

        resid_labelled_dict = {"nearest_resid": {"resname":
                {("labelled_shell"): {"shell_count": 0, "N_w": N_w,
                "donates_to": {"labelled_donators": 0,},
                "accepts_from": {"labelled_acceptors": 0,}
                }}}

        Get the orientational entropy of the molecules in this dict

        :param resid_labelled_dict: dictionary of format dict2 containing labelled
            coordination shells and HB donating and accepting
        """
        for resid, shell_label_key in sorted(list(resid_labelled_dict.items())):
            self.get_orientational_entropy_from_dict(
                shell_label_key, self.resid_labelled_Sorient[resid]
            )
            # get Nw, Nc Sor entropies via distribution of shell N occurance
            # No HB shell label info here
            self.get_orientational_entropy_from_Nx_dist(
                shell_label_key, self.resid_labelled_Sorient[resid]
            )
            # get SorHB using the Nc distribution method used above
            self.get_orientational_entropy_from_pbias_Nc_eff_dist(
                shell_label_key, self.resid_labelled_Sorient[resid]
            )

    def get_running_average(
        self, value: float, count: int, running_average_value: float, count_stored: int
    ):
        """
        For a given value, get it's running average from the current value

        :param value: the value that needs to be added to the running average
        :param count: the number of times the value occurs from statistics
        :param running_average_value: the currently stored running average
        :param count_stored: the currently stored count for the running average
        """
        new_count_stored = count_stored + count
        new_running_average = (
            value * count + running_average_value * count_stored
        ) / new_count_stored

        return new_running_average, new_count_stored

    def get_orientational_entropy_from_Nx_dist(
        self, labelled_dict: dict, Sorient_dict: dict
    ):
        # pylint: disable=too-many-locals
        """
        For a given dictionary containing labelled shells and HBing within the
        shell with format:

        labelled_dict = {"resname": {("labelled_shell"): {"shell_count": 0,
                                    "N_w": N_w,
                                    "donates_to": {"labelled_donators": 0,},
                                    "accepts_from": {"labelled_acceptors": 0,}
                                    }}}

        Get the orientational entropy of the molecules in this dict

        :param labelled_dict: dictionary of format dict1 containing labelled
            coordination shells and HB donating and accepting
        """
        Nc_counts, Nw_counts = self.get_Nx_dist(labelled_dict)

        for resname, N_c_key in Nc_counts.items():
            tot_count, Sorient_Nc_ave = 0, 0
            for N_c, count in N_c_key.items():
                if count > 0:  # sampling check
                    water = WaterOrientCalculator()
                    Sorient_Nc = water.get_non_HB_data(N_c)
                    # get running average for non-HB Sorient
                    Sorient_Nc_ave, tot_count = self.get_running_average(
                        Sorient_Nc, count, Sorient_Nc_ave, tot_count
                    )
            Sorient_dict[resname][9] = Sorient_Nc_ave

        for resname, N_w_key in Nw_counts.items():
            tot_count, Sorient_Nw_ave = 0, 0
            for N_w, count in N_w_key.items():
                if count > 0:  # sampling check
                    water = WaterOrientCalculator()
                    Sorient_Nw = water.get_non_HB_data(N_w)
                    # get running average for non-HB Sorient
                    Sorient_Nw_ave, tot_count = self.get_running_average(
                        Sorient_Nw, count, Sorient_Nw_ave, tot_count
                    )
            Sorient_dict[resname][10] = Sorient_Nw_ave

    def get_Nx_dist(self, labelled_dict: dict):
        # pylint: disable=too-many-locals
        """
        For a given dictionary containing labelled shells and HBing within the
        shell with format:

        labelled_dict = {"resname": {("labelled_shell"): {"shell_count": 0,
                                    "N_w": N_w,
                                    "donates_to": {"labelled_donators": 0,},
                                    "accepts_from": {"labelled_acceptors": 0,}
                                    }}}

        Get the orientational entropy of the molecules in this dict

        :param labelled_dict: dictionary of format dict1 containing labelled
            coordination shells and HB donating and accepting
        """
        Nc_counts = nested_dict()
        Nw_counts = nested_dict()
        for resname, shell_label_key in sorted(list(labelled_dict.items())):
            for shell_label, values in sorted(list(shell_label_key.items())):
                N_c = len(shell_label)
                if N_c not in Nc_counts[resname].keys():
                    Nc_counts[resname][N_c] = 0
                Nc_counts[resname][N_c] += 1

                N_w = values["N_w"]
                if N_w not in Nw_counts[resname].keys():
                    Nw_counts[resname][N_w] = 0
                Nw_counts[resname][N_w] += 1
        return Nc_counts, Nw_counts

    def get_orientational_entropy_from_pbias_Nc_eff_dist(
        self, labelled_dict: dict, Sorient_dict: dict
    ):
        # pylint: disable=too-many-locals
        """
        For a given dictionary containing labelled shells and HBing within the
        shell with format:

        labelled_dict = {"resname": {("labelled_shell"): {"shell_count": 0,
                                    "N_w": N_w,
                                    "donates_to": {"labelled_donators": 0,},
                                    "accepts_from": {"labelled_acceptors": 0,}
                                    }}}

        Get the orientational entropy of the molecules in this dict

        :param labelled_dict: dictionary of format dict1 containing labelled
            coordination shells and HB donating and accepting
        """
        Neff_HBbias_dict = self.get_pbias_Nc_eff_dist(labelled_dict)

        for resname, N_c_key in Neff_HBbias_dict.items():
            tot_count, Sorient_HB_ave = 0, 0
            for _N_c, info in N_c_key.items():
                Nc_eff = info["Nc_eff"]
                pbias = info["pbias"]
                count = info["count"]
                water = WaterOrientCalculator()
                Sorient_HB = water.get_orientation_S(Nc_eff, pbias)
                # get running average for non-HB Sorient
                Sorient_HB_ave, tot_count = self.get_running_average(
                    Sorient_HB, count, Sorient_HB_ave, tot_count
                )
            Sorient_dict[resname][8] = Sorient_HB_ave

    def get_pbias_Nc_eff_dist(self, labelled_dict: dict):
        """
        For a given dictionary containing labelled shells and HBing within the
        shell with format:

        labelled_dict = {"resname": {("labelled_shell"): {"shell_count": 0,
                                    "N_w": N_w,
                                    "donates_to": {"labelled_donators": 0,},
                                    "accepts_from": {"labelled_acceptors": 0,}
                                    }}}

        Get the orientational entropy of the molecules in this dict

        :param labelled_dict: dictionary of format dict1 containing labelled
            coordination shells and HB donating and accepting
        """
        Neff_HBbias_dict = (
            nested_dict()
        )  # dict for Neff and pbias_ave averaged over each Nc
        for resname, shell_label_key in sorted(list(labelled_dict.items())):
            for shell_label, values in sorted(list(shell_label_key.items())):
                if values["shell_count"] > 0:  # sampling check
                    N_c = len(shell_label)
                    # create object for Sorient
                    water = WaterOrientCalculator()
                    # calculate Sorient for water with given shell and solute
                    # neighbour using HB biasing
                    water.add_data(shell_label, values)

                    if N_c not in Neff_HBbias_dict[resname]:
                        Neff_HBbias_dict[resname][N_c] = {
                            "pbias": 0,
                            "Nc_eff": 0,
                            "count": 0,
                        }

                    (
                        Neff_HBbias_dict[resname][N_c]["Nc_eff"],
                        Neff_HBbias_dict[resname][N_c]["count"],
                    ) = self.get_running_average(
                        water.Nc_eff,
                        values["shell_count"],
                        Neff_HBbias_dict[resname][N_c]["Nc_eff"],
                        Neff_HBbias_dict[resname][N_c]["count"],
                    )
                    Neff_HBbias_dict[resname][N_c]["pbias"], _stored_shell_count = (
                        self.get_running_average(
                            water.pbias_ave,
                            values["shell_count"],
                            Neff_HBbias_dict[resname][N_c]["pbias"],
                            Neff_HBbias_dict[resname][N_c]["count"],
                        )
                    )
        return Neff_HBbias_dict

    def get_orientational_entropy_from_dict(
        self, labelled_dict: dict, Sorient_dict: dict
    ):
        # pylint: disable=too-many-locals
        """
        For a given dictionary containing labelled shells and HBing within the
        shell with format:

        labelled_dict = {"resname": {("labelled_shell"): {"shell_count": 0,
                                    "N_w": N_w,
                                    "donates_to": {"labelled_donators": 0,},
                                    "accepts_from": {"labelled_acceptors": 0,}
                                    }}}

        Get the orientational entropy of the molecules in this dict

        :param labelled_dict: dictionary of format dict1 containing labelled
            coordination shells and HB donating and accepting
        """
        for resname, shell_label_key in sorted(list(labelled_dict.items())):
            Sorient_ave, tot_count = 0, 0
            N_c_ave, N_w_ave, Nc_eff_ave, pbias_ave = 0, 0, 0, 0
            Sorient_Nc_ave, Sorient_Nw_ave = 0, 0
            for shell_label, values in sorted(list(shell_label_key.items())):
                if values["shell_count"] > 0:  # make sure enough sampling
                    # create object for Sorient
                    water = WaterOrientCalculator()
                    # calculate Sorient for water with given shell and solute
                    # neighbour using HB biasing
                    water.add_data(shell_label, values)

                    # add water entropies using previous methods that don't account
                    # for HBing
                    N_c = len(shell_label)
                    N_w = values["N_w"]
                    Sorient_Nc = water.get_non_HB_data(N_c)
                    Sorient_Nw = water.get_non_HB_data(N_w)

                    # only update tot_count here
                    # get running average for HB biased Sorient
                    Sorient_ave, tot_count = self.get_running_average(
                        water.Sorient, values["shell_count"], Sorient_ave, tot_count
                    )
                    # ignore the _tot_count
                    # get running average for non-HB Sorient
                    Sorient_Nc_ave, _tot_count = self.get_running_average(
                        Sorient_Nc, values["shell_count"], Sorient_Nc_ave, tot_count
                    )
                    Sorient_Nw_ave, _tot_count = self.get_running_average(
                        Sorient_Nw, values["shell_count"], Sorient_Nw_ave, tot_count
                    )
                    # get average terms for Nc and pbais
                    Nc_eff_ave, _tot_count = self.get_running_average(
                        water.Nc_eff, values["shell_count"], Nc_eff_ave, tot_count
                    )
                    pbias_ave, _tot_count = self.get_running_average(
                        water.pbias_ave, values["shell_count"], pbias_ave, tot_count
                    )
                    N_c_ave, _tot_count = self.get_running_average(
                        N_c, values["shell_count"], N_c_ave, tot_count
                    )
                    N_w_ave, _tot_count = self.get_running_average(
                        N_w, values["shell_count"], N_w_ave, tot_count
                    )
                Sorient_dict[resname] = [
                    Sorient_ave,
                    tot_count,
                    N_c_ave,
                    N_w_ave,
                    Nc_eff_ave,
                    pbias_ave,
                    Sorient_Nc_ave,
                    Sorient_Nw_ave,
                    0,  # new Sor HB, # calculated elsewhere
                    0,  # Sor Nc dist, # calculated elsewhere
                    0,  # Sor Nw dist, # calculated elsewhere
                ]


def print_Sorient_dicts(Sorient_dict: dict):
    # pylint: disable=too-many-locals
    """
    Print the orientational entropies of interfacial solvent

    :param Sorient_dict: dictionary containing orientational entropy values
    """
    terms = """
    Orientations
    ============
    resid: residue ID of the closest solute to the waters
    resname: name of the residue associated with the residue ID
    Sor: Orientational entropy of water around residue using HB biasing
    count: total number of waters around the residue across frames analysed
    N_c: average number of UAs around waters analysed
    N_w: average number of water UAs around waters analysed
    Nc_eff: average number of effective neighbouts that waters can HB with
    pbias: average of the probability of forming HBs with neighbouring UAs
    Sor_Nc: Orientational entropy of water around residue using Nc (no HB bias)
    Sor_Nw: Orientational entropy of water around residue using Nw (no HB bias)
    """
    print(textwrap.dedent(terms))

    print(
        "resid resname SorHB count N_c N_w Nc_eff pbias Sor_Nc Sor_Nw SorHB2 Sor_Nc2 Sor_Nw2"
    )
    for resid, resname_key in sorted(list(Sorient_dict.items())):
        for resname, [
            Sor,
            count,
            N_c,
            N_w,
            Nc_eff,
            pbias,
            Sor_Nc,
            Sor_Nw,
            Sor2,
            Sor_Nc2,
            Sor_Nw2,
        ] in sorted(list(resname_key.items())):
            decimals = 4
            print(
                f"{resid} {resname} {Sor:.{decimals}f} {count} "
                f"{N_c:.{decimals}f} {N_w:.{decimals}f} "
                f"{Nc_eff:.{decimals}f} {pbias:.{decimals}f} "
                f"{Sor_Nc:.{decimals}f} {Sor_Nw:.{decimals}f} "
                f"{Sor2:.{decimals}f} {Sor_Nc2:.{decimals}f} "
                f"{Sor_Nw2:.{decimals}f}"
            )
    print()
