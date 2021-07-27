from pathlib import Path
from enum import Enum
import argparse
from collections import namedtuple
import pandas as pd
import ROOT

GEMLayerId = namedtuple("GEMLayerId", ("region", "station", "layer"))

GEM_LAYER_ID_LIST = [
    GEMLayerId(1, 1, 2),
    GEMLayerId(1, 1, 1),
    GEMLayerId(-1, 1, 1),
    GEMLayerId(-1, 1, 2),
]

NUM_CHAMBERS = 36

CHAMBER_DATA_FIELDS = (
    "run",
    "region",
    "station",
    "layer",
    "chamber",
    "status",
    "total",
    "passed",
    "eff",
    "eff_low",
    "eff_up"
)

ChamberData = namedtuple("ChamberData", CHAMBER_DATA_FIELDS)

def parse_offline_dqmio_path(path):
    r"""
    DQM_V0001_R000342728__Cosmics__Commissioning2021-PromptReco-v1__DQMIO
    """
    dqm_ver_run, dataset, proc_ver, datatier = path.stem.split("__")
    prefix, dqm_ver, run = dqm_ver_run.split("_")
    assert prefix == "DQM"
    assert datatier == "DQMIO"
    assert run.startswith("R")
    run = int(run.lstrip("R"))
    return run

def parse_online_dqmio_path(path):
    r"""
    DQM_V0001_GEM_R000342728.root
    """
    prefix, dqm_ver, workspace, run = path.stem.split("_")
    assert prefix == "DQM"
    assert workspace == "GEM"
    assert run.startswith("R")
    run = int(run[1:])
    return run

class ReportStatus(Enum):
    NO_DATA = 0
    OK = 1
    ERROR = 2
    WARNING = 3



class ReportSummaryMap:

    def __init__(self,  hist):
        self._data = {}

        y_axis = hist.GetYaxis()
        for biny in range(1, y_axis.GetNbins() + 1):
            label = y_axis.GetBinLabel(biny)
            layer_id = self.paser_y_bin_label(label)

            report = [hist.GetBinContent(binx, biny) for binx in range(1, NUM_CHAMBERS + 1)]
            report = [ReportStatus(int(each)) for each in report]
            self._data[layer_id] = report
 
    def get_chamber_status(self, layer_id, chamber):
        return self._data[layer_id][chamber - 1]

    def paser_y_bin_label(self, label):
        r"""
        e.g. GE+1/1L1
        """
        if len(label) != 8:
            raise ValueError(label)
        _, _, region_sign, station, _, ring, _, layer = tuple(label)
        if region_sign == "+":
            region = 1
        elif region_sign == "-":
            region = -1
        else:
            raise ValueError

        try:
            station = int(station)
            ring = int(ring)
            layer = int(layer)
        except ValueError as error:
            print(label)
            raise error

        if ring != 1:
            raise ValueError(label)

        return GEMLayerId(region, station, layer)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--confidence-level", type=float, default=0.683)
    args = parser.parse_args()

    if not args.offline_dir.exists():
        raise FileNotFoundError(args.offline_dir.resolve())

    if not args.online_dir.exists():
        raise FileNotFoundError(args.online_dir.resolve())

    if args.output_path is None:
        args.output_path = args.offline_dir.name + ".csv"

    # NOTE
    online_report = {}
    for path in args.online_dir.glob("*.root"):
        run = parse_online_dqmio_path(path)
        root_file = ROOT.TFile(str(path))
        report_summary_map = root_file.Get(f"DQMData/Run {run}/GEM/Run summary/EventInfo/reportSummaryMap")
        online_report[run] = ReportSummaryMap(report_summary_map)

    # NOTE

    df = []
    for path in sorted(args.offline_dir.glob("*.root")):
        run = parse_offline_dqmio_path(path)
        root_file = ROOT.TFile(str(path))

        # use 2-leg STA
        gem_dir = root_file.Get(f"DQMData/Run {run}/GEM/Run summary/Efficiency/type1/Efficiency")

        for layer_id in GEM_LAYER_ID_LIST:
            region, station, layer = layer_id
            layer_name = f"GE{region * station:+d}1_L{layer}"

            h_total = gem_dir.Get(f"chamber_{layer_name}")
            h_passed = gem_dir.Get(f"chamber_{layer_name}_matched")

            for chamber in range(1, NUM_CHAMBERS + 1):
                status = online_report[run].get_chamber_status(layer_id, chamber)

                total = int(h_total.GetBinContent(chamber))
                passed = int(h_passed.GetBinContent(chamber))

                if status is ReportStatus.OK and total > 0:
                    eff = passed / total

                    lower_bound = ROOT.TEfficiency.ClopperPearson(
                        total, passed, args.confidence_level, False)

                    upper_bound = ROOT.TEfficiency.ClopperPearson(
                        total, passed, args.confidence_level, True)
                else:
                    eff = 0
                    lower_bound = 0
                    upper_bound = 0

                chamber_data = ChamberData(run, region, station, layer, chamber,
                                           status.value, total, passed, eff,
                                           lower_bound, upper_bound)
                df.append(chamber_data)

    df = pd.DataFrame(df)
    print(df.head())
    print("...")
    print(df.tail())

    df.to_csv(args.output_path, index=False)



if __name__ == "__main__":
    main()
