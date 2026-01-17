#!/usr/bin/env python3
"""
Convert Knowledge Graph from Entity Names to UMLS CUI Codes

This script converts a knowledge graph file with entity names to use UMLS CUI codes
using a mapping file.

Input files:
  - kg_clean.txt: Knowledge graph with entity names (entity1,relation,entity2)
  - umls_mapping_triples.txt: Entity to CUI mapping (entity_name|mapped_to_cui|CUI_CODE)

Output:
  - kg_cui.txt: Knowledge graph with CUI codes (CUI1,relation,CUI2)

Usage:
    python convert_names_to_cui.py --kg kg_clean.txt --mapping umls_mapping_triples.txt --output kg_cui.txt
"""

import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple
import re


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for matching.

    Converts to lowercase and removes extra whitespace.
    """
    # Convert to lowercase
    name = name.lower().strip()
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    return name


def load_cui_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load entity to CUI mapping from file.

    Args:
        mapping_file: Path to umls_mapping_triples.txt

    Returns:
        Dictionary mapping normalized entity names to CUI codes
    """
    mapping = {}
    unmapped_count = 0

    print(f"Loading CUI mapping from {mapping_file}...")

    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse line: entity_name|mapped_to_cui|CUI_CODE
            parts = line.split('|')
            if len(parts) != 3:
                print(f"Warning: Line {line_num} has invalid format: {line}")
                continue

            entity_name, relation, cui_code = parts

            # Normalize entity name for matching
            normalized_name = normalize_entity_name(entity_name)

            # Store mapping
            if normalized_name in mapping and mapping[normalized_name] != cui_code:
                print(f"Warning: Duplicate mapping for '{entity_name}': {mapping[normalized_name]} vs {cui_code}")

            mapping[normalized_name] = cui_code

    print(f"✓ Loaded {len(mapping)} entity-to-CUI mappings")

    return mapping


def load_knowledge_graph(kg_file: str) -> List[Tuple[str, str, str]]:
    """
    Load knowledge graph triples from file.

    Args:
        kg_file: Path to kg_clean.txt

    Returns:
        List of (head, relation, tail) tuples
    """
    triples = []

    print(f"\nLoading knowledge graph from {kg_file}...")

    with open(kg_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse line: entity1,relation,entity2
            parts = line.split(',')
            if len(parts) < 3:
                print(f"Warning: Line {line_num} has invalid format: {line}")
                continue

            # Handle case where relation might contain commas
            if len(parts) > 3:
                # Assume format: entity1,relation_with_commas,entity2
                head = parts[0]
                tail = parts[-1]
                relation = ','.join(parts[1:-1])
            else:
                head, relation, tail = parts

            triples.append((head.strip(), relation.strip(), tail.strip()))

    print(f"✓ Loaded {len(triples)} triples")

    return triples


def convert_to_cui(
    triples: List[Tuple[str, str, str]],
    cui_mapping: Dict[str, str],
    allow_unmapped: bool = False
) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
    """
    Convert knowledge graph triples from entity names to CUI codes.

    Args:
        triples: List of (head, relation, tail) tuples with entity names
        cui_mapping: Dictionary mapping entity names to CUI codes
        allow_unmapped: If True, keep unmapped entities as-is; if False, skip them

    Returns:
        Tuple of (converted_triples, unmapped_entities)
    """
    converted = []
    unmapped_entities = set()
    skipped_count = 0

    print(f"\nConverting entities to CUI codes...")

    for head, relation, tail in triples:
        # Normalize for lookup
        head_norm = normalize_entity_name(head)
        tail_norm = normalize_entity_name(tail)

        # Look up CUI codes
        head_cui = cui_mapping.get(head_norm)
        tail_cui = cui_mapping.get(tail_norm)

        # Track unmapped entities
        if head_cui is None:
            unmapped_entities.add(head)
        if tail_cui is None:
            unmapped_entities.add(tail)

        # Decide what to do with unmapped entities
        if head_cui is None or tail_cui is None:
            if allow_unmapped:
                # Keep original names for unmapped entities
                head_final = head_cui if head_cui else head
                tail_final = tail_cui if tail_cui else tail
                converted.append((head_final, relation, tail_final))
            else:
                # Skip triple if any entity is unmapped
                skipped_count += 1
                continue
        else:
            converted.append((head_cui, relation, tail_cui))

    print(f"✓ Converted {len(converted)} triples")
    if skipped_count > 0:
        print(f"✗ Skipped {skipped_count} triples due to unmapped entities")
    if len(unmapped_entities) > 0:
        print(f"⚠ Found {len(unmapped_entities)} unmapped entities")

    return converted, unmapped_entities


def save_cui_graph(triples: List[Tuple[str, str, str]], output_file: str):
    """
    Save CUI-based knowledge graph to file.

    Args:
        triples: List of (head_cui, relation, tail_cui) tuples
        output_file: Path to output file
    """
    print(f"\nSaving CUI-based knowledge graph to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("head,relation,tail\n")

        # Write triples
        for head, relation, tail in triples:
            f.write(f"{head},{relation},{tail}\n")

    print(f"✓ Saved {len(triples)} triples to {output_file}")


def save_unmapped_report(unmapped_entities: Set[str], output_file: str):
    """Save report of unmapped entities."""
    if len(unmapped_entities) == 0:
        print("✓ All entities were mapped successfully!")
        return

    print(f"\nSaving unmapped entities report to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Unmapped Entities\n")
        f.write("=" * 60 + "\n\n")

        for entity in sorted(unmapped_entities):
            f.write(f"{entity}\n")

    print(f"✓ Saved {len(unmapped_entities)} unmapped entities to {output_file}")


def print_statistics(original_triples: List, converted_triples: List, unmapped_entities: Set):
    """Print conversion statistics."""
    print("\n" + "=" * 70)
    print("Conversion Statistics")
    print("=" * 70)
    print(f"Original triples:      {len(original_triples)}")
    print(f"Converted triples:     {len(converted_triples)}")
    print(f"Skipped triples:       {len(original_triples) - len(converted_triples)}")
    print(f"Unmapped entities:     {len(unmapped_entities)}")

    if len(converted_triples) > 0:
        success_rate = (len(converted_triples) / len(original_triples)) * 100
        print(f"Success rate:          {success_rate:.1f}%")

    # Extract unique entities and relations
    entities = set()
    relations = set()
    for h, r, t in converted_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    print(f"\nConverted graph:")
    print(f"  Unique entities:     {len(entities)}")
    print(f"  Unique relations:    {len(relations)}")
    print(f"  Total triples:       {len(converted_triples)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Convert knowledge graph from entity names to UMLS CUI codes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--kg', type=str, required=True,
                       help='Input knowledge graph file (kg_clean.txt)')
    parser.add_argument('--mapping', type=str, required=True,
                       help='UMLS mapping file (umls_mapping_triples.txt)')
    parser.add_argument('--output', type=str, default='kg_cui.txt',
                       help='Output file for CUI-based graph')
    parser.add_argument('--allow-unmapped', action='store_true',
                       help='Keep unmapped entities as-is (default: skip triples with unmapped entities)')
    parser.add_argument('--report-unmapped', type=str, default='unmapped_entities.txt',
                       help='File to save list of unmapped entities')

    args = parser.parse_args()

    print("=" * 70)
    print("Knowledge Graph Entity Name → UMLS CUI Converter")
    print("=" * 70)
    print(f"KG file:       {args.kg}")
    print(f"Mapping file:  {args.mapping}")
    print(f"Output file:   {args.output}")
    print(f"Allow unmapped: {args.allow_unmapped}")
    print("=" * 70)

    # Step 1: Load CUI mapping
    cui_mapping = load_cui_mapping(args.mapping)

    # Step 2: Load knowledge graph
    triples = load_knowledge_graph(args.kg)

    # Step 3: Convert to CUI codes
    converted_triples, unmapped_entities = convert_to_cui(
        triples,
        cui_mapping,
        allow_unmapped=args.allow_unmapped
    )

    # Step 4: Save converted graph
    save_cui_graph(converted_triples, args.output)

    # Step 5: Save unmapped entities report
    if len(unmapped_entities) > 0:
        save_unmapped_report(unmapped_entities, args.report_unmapped)

    # Step 6: Print statistics
    print_statistics(triples, converted_triples, unmapped_entities)

    # Final message
    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Review unmapped entities (if any): {args.report_unmapped}")
    print(f"2. Convert to FuseLinker format:")
    print(f"   python convert_to_fuselinker_format.py \\")
    print(f"       --input {args.output} \\")
    print(f"       --output mybkg_cui")
    print(f"3. Train model:")
    print(f"   cd fuselinker-complex")
    print(f"   python main.py --data mybkg_cui --text_embedding_file sapbert_embeddings ...")
    print("=" * 70)


if __name__ == '__main__':
    main()
