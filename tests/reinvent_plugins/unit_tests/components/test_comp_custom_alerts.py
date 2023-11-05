import numpy as np

from reinvent_plugins.components.comp_custom_alerts import CustomAlerts
from reinvent_plugins.components.comp_custom_alerts import Parameters


def test_comp_custom_alerts():
    endpoint1_smarts = [
        "[*;r8]",
        "[*;r9]",
        "[*;r10]",
        "[*;r11]",
        "[*;r12]",
        "[*;r13]",
        "[*;r14]",
        "[*;r15]",
        "[*;r16]",
        "[*;r17]",
        "[#8][#8]",
        "[#6;+]",
        "[#16][#16]",
        "[#7;!n][S;!$(S(=O)=O)]",
        "[#7;!n][#7;!n]",
        "C#C",
        "C(=[O,S])[O,S]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
        "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
        "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
        "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
        "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
    ]

    # All attributes in Parameters always expect a list of parameters
    params = Parameters(smarts=[endpoint1_smarts])
    component = CustomAlerts(params)
    results = component(["c1ccccc1", "CCC#CCC"])

    assert (results.scores == np.array([1, 0])).all()
