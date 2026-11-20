"""Running-average weighting of the orientational entropy summary fields."""

import pytest

from waterEntropy.entropy.orientations import Orientations, WaterOrientCalculator
from waterEntropy.utils.helpers import nested_dict

FIELDS = (
    "Sorient_ave",
    "tot_count",
    "N_c_ave",
    "N_w_ave",
    "Nc_eff_ave",
    "pbias_ave",
    "Sorient_Nc_ave",
    "Sorient_Nw_ave",
)


def _labelled(shells):
    """Build {resname: {shell_label: values}} with no hydrogen bonding recorded."""
    labelled = nested_dict()
    for label, count, n_w in shells:
        labelled["WAT"][tuple(sorted(label))] = {
            "shell_count": count,
            "N_w": n_w,
            "donates_to": {},
            "accepts_from": {},
        }
    return labelled


def _summary(shells):
    out = nested_dict()
    Orientations().get_orientational_entropy_from_dict(_labelled(shells), out)
    return out["WAT"]


def test_single_shell_type_fields_equal_their_values():
    s = _summary([(("WAT",) * 4, 7, 3)])
    non_hb = WaterOrientCalculator().get_non_HB_data(4)

    assert s["tot_count"] == 7
    assert s["N_c_ave"] == pytest.approx(4.0)
    assert s["N_w_ave"] == pytest.approx(3.0)
    assert s["Sorient_Nc_ave"] == pytest.approx(non_hb)


def test_multiple_shell_types_use_count_weighted_means():
    shells = [(("WAT",) * 4, 10, 4), (("WAT",) * 6, 30, 5)]
    s = _summary(shells)
    calc = WaterOrientCalculator()

    def weighted(values):
        return sum(v * c for v, (_, c, _) in zip(values, shells, strict=True)) / 40

    assert s["tot_count"] == 40
    assert s["N_c_ave"] == pytest.approx(weighted([4, 6]))
    assert s["N_w_ave"] == pytest.approx(weighted([4, 5]))
    assert s["Sorient_Nc_ave"] == pytest.approx(
        weighted([calc.get_non_HB_data(4), calc.get_non_HB_data(6)])
    )
    assert s["Sorient_Nw_ave"] == pytest.approx(
        weighted([calc.get_non_HB_data(4), calc.get_non_HB_data(5)])
    )
