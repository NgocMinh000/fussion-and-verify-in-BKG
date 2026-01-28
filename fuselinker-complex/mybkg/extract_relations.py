import csv

input_path = "kg_cui.txt"
output_path = "relations_unique.txt"   # hoặc .csv nếu bạn muốn

relations = set()

with open(input_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rel = (row.get("relation") or "").strip()
        if rel:
            relations.add(rel)

# Ghi ra file (mỗi relation 1 dòng), sắp xếp cho dễ nhìn
with open(output_path, "w", encoding="utf-8", newline="") as f:
    for rel in sorted(relations):
        f.write(rel + "\n")

print(f"Done. Wrote {len(relations)} unique relations to {output_path}")
