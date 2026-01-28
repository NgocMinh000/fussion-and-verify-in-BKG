# map_relations_to_txt_3col.py
import csv
import argparse
from pathlib import Path


def load_mapping(mapping_path: str, key_col: str = "relation_raw", val_col: str = "category_50") -> dict:
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or key_col not in reader.fieldnames or val_col not in reader.fieldnames:
            raise ValueError(
                f"Mapping file must contain columns '{key_col}' and '{val_col}'. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            k = (row.get(key_col) or "").strip()
            v = (row.get(val_col) or "").strip()
            if k:
                mapping[k] = v
    return mapping


def map_kg_to_txt_3col(input_kg: str, output_txt: str, mapping: dict, on_missing: str = "keep"):
    """
    Input KG: CSV with header head,relation,tail
    Output: .txt text file with header head,relation,tail
            relation column replaced by mapped value
    """
    total = mapped = missing = dropped = 0

    with open(input_kg, "r", encoding="utf-8", newline="") as fin, open(
        output_txt, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        required = {"head", "relation", "tail"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"KG file must have header with columns head,relation,tail. Found: {reader.fieldnames}"
            )

        # Header giống file gốc
        fout.write("head,relation,tail\n")

        for row in reader:
            total += 1
            h = (row.get("head") or "").strip()
            r = (row.get("relation") or "").strip()
            t = (row.get("tail") or "").strip()

            if not (h and r and t):
                dropped += 1
                continue

            new_r = mapping.get(r)
            if not new_r:
                missing += 1
                if on_missing == "drop":
                    dropped += 1
                    continue
                elif on_missing == "unmapped":
                    new_r = "UNMAPPED"
                else:  # keep
                    new_r = r
            else:
                mapped += 1

            fout.write(f"{h},{new_r},{t}\n")

    print("Done.")
    print(f"Total triples read: {total}")
    print(f"Mapped: {mapped}")
    print(f"Missing mapping: {missing}")
    print(f"Dropped: {dropped}")
    print(f"Output: {output_txt}")


def main():
    ap = argparse.ArgumentParser(description="Map KG relations and export .txt with head,relation,tail.")
    ap.add_argument("--kg", default="kg_cui.txt", help="Input KG file (CSV header head,relation,tail)")
    ap.add_argument("--map", dest="mapfile", default="relations_mapped_50_simple.csv",
                    help="Mapping CSV (relations_mapped_50_simple.csv)")
    ap.add_argument("--val-col", default="category_50",
                    help="Mapped column to use: category_50 or category_simple")
    ap.add_argument("--out", default="kg_cui_mapped.txt", help="Output .txt path")
    ap.add_argument("--on-missing", choices=["keep", "unmapped", "drop"], default="keep",
                    help="What to do if relation not in mapping")
    args = ap.parse_args()

    if not Path(args.kg).exists():
        raise FileNotFoundError(f"KG file not found: {args.kg}")
    if not Path(args.mapfile).exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapfile}")

    mapping = load_mapping(args.mapfile, key_col="relation_raw", val_col=args.val_col)
    map_kg_to_txt_3col(args.kg, args.out, mapping, on_missing=args.on_missing)


if __name__ == "__main__":
    main()
