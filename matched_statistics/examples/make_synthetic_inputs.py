"""Create artificial CSV fixtures only. No study data are read."""
import csv
import json
from importlib.resources import files
from pathlib import Path
import sys


def main():
    target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True)
    plans=json.loads(files("paper2_statistics").joinpath("plans.json").read_text(encoding="utf-8"))
    systems=plans["selected"]["systems"]
    for index,name in enumerate(systems):
        path=target/(name+".csv")
        with path.open("x",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["roi_id","patient_id","outer_fold","y_true","y_pred"])
            for fold in range(5):
                for roi in range(1+fold%3):
                    grade=(fold+roi)%4
                    writer.writerow([f"ARTIFICIAL_ROI_{fold}_{roi}",f"ARTIFICIAL_PATIENT_{fold}",fold,grade,(grade+index%3)%4])
    for name,plan in plans.items():
        with (target/(name+".json")).open("x",encoding="utf-8") as stream:
            json.dump({"predictions":{system:system+".csv" for system in plan["systems"]}},stream,indent=2)


if __name__=="__main__":main()
