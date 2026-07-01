"""
gen_split_csv_v2.py — Generate train_test_split.csv for data_multislice_v2.

Takes 5% of patients from each source (DUKE, ISPY1, ISPY2, YUNNAN, NACT)
as test, the rest as train. Split is done at patient level.

Usage:
    python gen_split_csv_v2.py \
        --data_dir /path/to/data_multislice_v2 \
        --test_ratio 0.05 \
        --seed 42
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def get_patient_id(filename: str) -> str:
    """Extract patient_id from filename like DUKE_001_p1.mha → DUKE_001."""
    stem = filename.replace(".mha", "")
    if "_s+" in stem or "_s-" in stem:
        stem = stem.rsplit("_s", 1)[0]
    if "_p" in stem:
        stem = stem.rsplit("_p", 1)[0]
    return stem


def get_source(patient_id: str) -> str:
    """Determine source from patient_id prefix."""
    if patient_id.startswith("DUKE"):
        return "DUKE"
    elif patient_id.startswith("ISPY1"):
        return "ISPY1"
    elif patient_id.startswith("ISPY2"):
        return "ISPY2"
    elif patient_id.startswith("YUNNAN"):
        return "YUNNAN"
    elif patient_id.startswith("NACT"):
        return "NACT"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to data_multislice_v2")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Fraction of patients for test per source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output CSV path (default: data_dir/train_test_split.csv)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output) if args.output else data_dir / "train_test_split.csv"

    # Scan input directory
    input_dir = data_dir / "mha" / "input"
    if not input_dir.exists():
        print(f"ERROR: {input_dir} does not exist")
        return

    files = sorted(f.name for f in input_dir.iterdir() if f.suffix == ".mha")

    # Group patients by source
    patients_by_source = defaultdict(set)
    for fname in files:
        pid = get_patient_id(fname)
        source = get_source(pid)
        patients_by_source[source].add(pid)

    # Select 5% test patients from each source
    random.seed(args.seed)
    test_patients = set()

    print(f"Splitting with test_ratio={args.test_ratio}, seed={args.seed}")
    print(f"{'Source':<10} {'Total':<8} {'Test':<8} {'Train':<8}")
    print("-" * 36)

    for source in sorted(patients_by_source.keys()):
        patients = sorted(patients_by_source[source])
        n_test = max(1, int(len(patients) * args.test_ratio))
        random.shuffle(patients)
        test_set = set(patients[:n_test])
        test_patients.update(test_set)
        print(f"{source:<10} {len(patients):<8} {n_test:<8} {len(patients) - n_test:<8}")

    # Generate CSV
    rows = []
    for fname in files:
        case_id = fname.replace(".mha", "")
        pid = get_patient_id(fname)
        source = get_source(pid)
        split = "test" if pid in test_patients else "train"
        rows.append({
            "case_id": case_id,
            "patient_id": pid,
            "source": source,
            "split": split,
        })

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "patient_id", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    splits = defaultdict(int)
    for r in rows:
        splits[r["split"]] += 1

    print(f"\nGenerated: {output}")
    print(f"  Total cases: {len(rows)}")
    print(f"  Train: {splits['train']}")
    print(f"  Test:  {splits['test']}")


if __name__ == "__main__":
    main()
