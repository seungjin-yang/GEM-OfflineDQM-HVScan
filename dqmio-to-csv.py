from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import re
import argparse
import warnings
import yaml
import pandas as pd
import ROOT

NUM_CHAMBERS = 36 # consider only GE11 for now

LAYER_LABEL_PATTERN = re.compile("^GE[0-2]1-(M|P)-L[1-6]$")
LAYER_ONLINE_LABEL_PATTERN = re.compile("^GE(\+|\-)[0-2]\/1L[1-6]$")
LAYER_OFFLINE_LABEL_PATTERN = re.compile("^GE(\+|\-)[0-2]1_L[1-6]$")
ME_PATTERN = re.compile("^chamber_GE(\+|\-)[0-2]1_L[1-6]$")

@dataclass(frozen=True, eq=True)
class GEMLayerId:
    region: int
    station: int
    layer: int

    @classmethod
    def from_label(cls, label):
        r"""
        e.g. GE11-P-L1
        """
        assert LAYER_LABEL_PATTERN.match(label), label
        region = 1 if label[5] == "P" else -1
        station = int(label[2])
        layer = int(label[-1])
        return cls(region, station, layer)

    @classmethod
    def from_online_label(cls, label):
        r"""
        e.g. GE+1/1L1
        """
        assert LAYER_ONLINE_LABEL_PATTERN.match(label), label
        region = 1 if label[2] == "+" else -1
        station = int(label[3])
        layer = int(label[-1])
        return cls(region, station, layer)

    @classmethod
    def from_offline_label(cls, label):
        r"""actually name, not label
        e.g. GE+11_L1
        """
        assert LAYER_OFFLINE_LABEL_PATTERN.match(label), label
        region = 1 if label[2] == "+" else -1
        station = int(label[3])
        layer = int(label[-1])
        return cls(region, station, layer)

    @property
    def region_sign(self):
        if self.region == 1:
            return "P"
        elif self.region == -1:
            return "M"
        else:
            raise ValueError(f"region={self.region}")

    @property
    def label(self):
        return f"GE{self.station}1-{self.region_sign}-L{self.layer}"

    @property
    def offline_label(self):
        return f"GE{self.region * self.station:+d}1_L{self.layer}"



def load_yaml(path):
    with open(path, "r") as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return data


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
            layer_id = GEMLayerId.from_online_label(label)

            report = [hist.GetBinContent(binx, biny) for binx in range(1, NUM_CHAMBERS + 1)]
            report = [ReportStatus(int(each)) for each in report]

            self._data[layer_id] = report
 
    def get_status(self, layer_id, chamber):
        return self._data[layer_id][chamber - 1]


class BadChamberListReader:
    def __init__(self, path):
        bad_chamber_data = load_yaml(path)

        self._bad_chamber_data = {}
        for run, layer_dict in bad_chamber_data.items():
            self._bad_chamber_data[run] = {}
            for layer_label, bad_chamber_list in layer_dict.items():
                layer_id = GEMLayerId.from_label(layer_label)
                self._bad_chamber_data[run][layer_id] = bad_chamber_list

    def get_dc_report(self, run, layer_id, chamber):
        r"""
        """
        if run in self._bad_chamber_data:
            has_dc = True
            is_good = chamber not in self._bad_chamber_data[run][layer_id]
        else:
            has_dc = False
            is_good = False
        return has_dc, is_good


class OMSReader:
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def get_duration(self, run):
        duration = self.df[self.df.run == run].duration.to_list()[0]
        duration = pd.to_datetime(duration, format="%H:%M:%S")
        duration = duration.hour + duration.minute / 60 + duration.second / 3600
        return duration


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--offline", type=Path, required=True,
                        help="Directory to read offline DQMIO files")
    parser.add_argument("--online", type=Path, required=True,
                        help="Directory to read onfline DQMIO files")
    parser.add_argument("--output", type=Path, default="./",
                        help="Directory to write data")
    parser.add_argument("--confidence-level", type=float, default=0.683,
                        help="Confidence level for Clopper-Pearson function")
    args = parser.parse_args()

    if not args.offline.exists():
        raise FileNotFoundError(args.offline.resolve())

    if not args.online.exists():
        raise FileNotFoundError(args.online.resolve())

    if not args.output.exists():
        raise FileNotFoundError(args.output.resolve())

    # XXX
    hv_run_list = load_yaml("./data/hv-run.yaml")
    # {hv: run_list} to {run: hv}
    run2hv = {}
    for hv, run_list in hv_run_list.items():
        run2hv.update({run: hv for run in run_list})

    # XXX OMS
    oms = OMSReader("./data/oms.csv")

    # XXX Online DQM
    online_report = {}
    for path in args.online.glob("*.root"):
        run = parse_online_dqmio_path(path)
        root_file = ROOT.TFile(str(path))
        report_summary_map = root_file.Get(
            f"DQMData/Run {run}/GEM/Run summary/EventInfo/reportSummaryMap")
        online_report[run] = ReportSummaryMap(report_summary_map)

    # XXX Bad Chamber
    bad_chamber_data = BadChamberListReader("./data/bad-chamber.yaml")

    # XXX Offline DQM
    df = []
    for path in sorted(args.offline.glob("*.root")):
        run = parse_offline_dqmio_path(path)
        duration = oms.get_duration(run)

        root_file = ROOT.TFile(str(path))
        # use 2-leg STA
        gem_dir = root_file.Get(
            f"DQMData/Run {run}/GEM/Run summary/Efficiency/type1/Efficiency")
        for key in gem_dir.GetListOfKeys():
            key = key.GetName()
            if ME_PATTERN.match(key) is None:
                continue

            h_total = gem_dir.Get(key)
            h_passed = gem_dir.Get(key + "_matched")

            layer_label = key[len("chamber_"): ]
            layer_id = GEMLayerId.from_offline_label(layer_label)

            for chamber in range(1, NUM_CHAMBERS + 1):
                status = online_report[run].get_status(layer_id, chamber)
                has_dc, is_good = bad_chamber_data.get_dc_report(run, layer_id,
                                                                 chamber)

                total = int(h_total.GetBinContent(chamber))
                passed = int(h_passed.GetBinContent(chamber))

                if total > 0:
                    eff = passed / total

                    lower_bound = ROOT.TEfficiency.ClopperPearson(
                        total, passed, args.confidence_level, False)

                    upper_bound = ROOT.TEfficiency.ClopperPearson(
                        total, passed, args.confidence_level, True)
                else:
                    eff = 0
                    lower_bound = 0
                    upper_bound = 0

                row = {
                    "run": run,
                    "duration": duration,
                    "hv": run2hv[run],
                    "region": layer_id.region,
                    "station": layer_id.station,
                    "layer": layer_id.layer,
                    "chamber": chamber,
                    "online_report": status.value,
                    "has_dc": has_dc,
                    "is_good": is_good,
                    "total": total,
                    "passed": passed,
                    "eff": eff,
                    "eff_low": lower_bound,
                    "eff_up": upper_bound,
                }
                df.append(row)

    df = pd.DataFrame(df)
    print(df.head())
    print("...")
    print(df.tail())

    output_path = args.output.joinpath(args.offline.name).with_suffix(".csv")
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    main()
