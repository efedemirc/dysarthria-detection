"""Assemble per-utterance JSON records into features_torgo.csv.

Reads the JSON records written by build_features.py, aligns their feature keys
into a single column union (missing -> NaN), joins the documented severity map
for dysarthric speakers, and writes a tidy CSV for the modelling harness.

Columns: source_id, speaker_id, group, label_detection, label_severity,
         <one column per acoustic feature>.

Severity map provenance
-----------------------
TORGO does not ship a single machine-readable 3-class severity column; severity
is documented per speaker in the database literature (Rudzicz et al., 2012) from
Frenchay/intelligibility assessment. The mapping below is applied ONLY to
dysarthric speakers and is kept here, explicit and editable, so its provenance
is auditable. CONFIRM these against the TORGO documentation before reporting
severity results in the thesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Documented TORGO dysarthria severity (Frenchay Dysarthria Assessment), per
# Rudzicz et al. (2012): F01/M01/M02/M04 are SEVERE, M05 is moderately-to-
# severely dysarthric, F03 is MODERATE, and M03/F04 are "very mild". The
# standard 3-class binning used across the TORGO severity literature is applied:
# M05 (moderate-to-severe) -> moderate; M03/F04 (very mild) -> mild. Verified
# against the TORGO documentation 2026-06-22.
SEVERITY_MAP = {
    "M01": "severe", "M02": "severe", "M03": "mild",
    "M04": "severe", "M05": "moderate",
    "F01": "severe", "F03": "moderate", "F04": "mild",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for jp in sorted(Path(args.records_dir).glob("*.json")):
        try:
            obj = json.loads(jp.read_text())
        except Exception:
            continue
        spk = obj["speaker_id"]
        group = obj["group"]
        row = {
            "source_id": obj["source_id"],
            "speaker_id": spk,
            "group": group,
            "label_detection": "dysarthric" if group == "dysarthric" else "control",
            "label_severity": ("control" if group == "control"
                               else SEVERITY_MAP.get(spk, "unknown")),
        }
        row.update(obj.get("features", {}))
        rows.append(row)

    if not rows:
        print("no records found")
        return 1

    df = pd.DataFrame(rows)
    meta = ["source_id", "speaker_id", "group", "label_detection", "label_severity"]
    feat_cols = [c for c in df.columns if c not in meta]
    df = df[meta + sorted(feat_cols)]
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows x {len(feat_cols)} features")
    print("speakers:", df["speaker_id"].value_counts().to_dict())
    print("detection labels:", df["label_detection"].value_counts().to_dict())
    print("severity labels:", df["label_severity"].value_counts().to_dict())
    # report feature completeness
    na = df[feat_cols].isna().mean().sort_values(ascending=False)
    incomplete = na[na > 0]
    if len(incomplete):
        print("features with missing values (frac):",
              {k: round(v, 2) for k, v in incomplete.head(10).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
