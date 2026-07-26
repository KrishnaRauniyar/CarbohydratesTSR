#!/usr/bin/env python3
"""Create CI/WCI/NCI labels from atom-level carbohydrate-protein distances."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


OUTPUT_COLUMNS = ["protein", "chain", "carb_name", "carb_res", "group"]


@dataclass(frozen=True)
class AtomId:
    pdb_id: str
    chain: str
    residue_name: str
    residue_number: str
    atom_name: str


@dataclass
class Interaction:
    protein: str
    chain: str
    carb_name: str
    carb_res: str
    min_distance: float


def parse_atom_id(value: str) -> AtomId:
    """Parse ids like 7L8Y_G_NAG_1_C1 or 7L8Y_C_ASN_392_ND2."""
    parts = value.strip().split("_")
    if len(parts) < 5:
        raise ValueError(f"Could not parse atom id: {value!r}")

    return AtomId(
        pdb_id="_".join(parts[:-4]),
        chain=parts[-4],
        residue_name=parts[-3],
        residue_number=parts[-2],
        atom_name=parts[-1],
    )


def resolve_column(
    fieldnames: list[str],
    explicit_name: str | None,
    default_name: str | None,
    role: str,
) -> str:
    if explicit_name:
        matches = [name for name in fieldnames if name.lower() == explicit_name.lower()]
        if not matches:
            raise ValueError(
                f"Requested {role} column {explicit_name!r} was not found. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return matches[0]

    if default_name and default_name in fieldnames:
        return default_name

    matches = [name for name in fieldnames if role.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]

    if role == "distance":
        distance_matches = [name for name in fieldnames if "distance" in name.lower()]
        if len(distance_matches) == 1:
            return distance_matches[0]
        return fieldnames[-1]

    raise ValueError(
        f"Could not identify the {role} column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def label_from_distance(
    distance: float,
    ci_cutoff: float,
    wci_cutoff: float,
    ci_label: str,
    wci_label: str,
    nci_label: str,
) -> str:
    if distance <= ci_cutoff:
        return ci_label
    if distance <= wci_cutoff:
        return wci_label
    return nci_label


def build_interactions(
    input_csv: Path,
    carb_name: str,
    drug_column_name: str | None,
    distance_column_name: str | None,
) -> OrderedDict[tuple[str, str, str, str], Interaction]:
    interactions: OrderedDict[tuple[str, str, str, str], Interaction] = OrderedDict()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")

        drug_column = resolve_column(reader.fieldnames, drug_column_name, "drug", "drug")
        distance_column = resolve_column(
            reader.fieldnames, distance_column_name, None, "distance"
        )

        target_carb_name = carb_name.upper()
        for line_number, row in enumerate(reader, start=2):
            drug_atom = parse_atom_id(row[drug_column])
            if drug_atom.residue_name.upper() != target_carb_name:
                continue

            try:
                distance = float(row[distance_column])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid distance on line {line_number}: {row[distance_column]!r}"
                ) from exc

            key = (
                drug_atom.pdb_id,
                drug_atom.chain,
                drug_atom.residue_name,
                drug_atom.residue_number,
            )

            existing = interactions.get(key)
            if existing is None:
                interactions[key] = Interaction(
                    protein=drug_atom.pdb_id,
                    chain=drug_atom.chain,
                    carb_name=drug_atom.residue_name,
                    carb_res=drug_atom.residue_number,
                    min_distance=distance,
                )
            elif distance < existing.min_distance:
                existing.min_distance = distance

    return interactions


def write_group_csv(
    interactions: OrderedDict[tuple[str, str, str, str], Interaction],
    output_csv: Path,
    ci_cutoff: float,
    wci_cutoff: float,
    ci_label: str,
    wci_label: str,
    nci_label: str,
) -> dict[str, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = {ci_label: 0, wci_label: 0, nci_label: 0}

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for interaction in interactions.values():
            group = label_from_distance(
                interaction.min_distance,
                ci_cutoff=ci_cutoff,
                wci_cutoff=wci_cutoff,
                ci_label=ci_label,
                wci_label=wci_label,
                nci_label=nci_label,
            )
            counts[group] = counts.get(group, 0) + 1
            writer.writerow(
                {
                    "protein": interaction.protein,
                    "chain": interaction.chain,
                    "carb_name": interaction.carb_name,
                    "carb_res": interaction.carb_res,
                    "group": group,
                }
            )

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read atom-level cross-distance rows, keep one carbohydrate residue "
            "name, and write PD1-style CI/WCI/NCI interaction labels."
        )
    )
    parser.add_argument(
        "-i",
        "--input-csv",
        required=True,
        type=Path,
        help="Atom-level cross-distance CSV, for example carbsdrug_gp120_cross.csv.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV with columns: protein, chain, carb_name, carb_res, group.",
    )
    parser.add_argument(
        "--carb-name",
        "--ligand-name",
        default="NAG",
        help="Carbohydrate residue name to keep from the first column. Default: NAG.",
    )
    parser.add_argument(
        "--ci-cutoff",
        type=float,
        default=1.5,
        help="Minimum distance <= this value is labeled CI. Default: 1.5.",
    )
    parser.add_argument(
        "--wci-cutoff",
        "--nci-cutoff",
        dest="wci_cutoff",
        type=float,
        default=2.0,
        help=(
            "Minimum distance > CI cutoff and <= this value is WCI; "
            "distance above it is NCI. Default: 2.0."
        ),
    )
    parser.add_argument("--ci-label", default="ci", help="Label for CI rows.")
    parser.add_argument("--wci-label", default="wci", help="Label for WCI rows.")
    parser.add_argument("--nci-label", default="nci", help="Label for NCI rows.")
    parser.add_argument(
        "--drug-column",
        help="Column containing carbohydrate atom ids. Default: drug.",
    )
    parser.add_argument(
        "--protein-column",
        help=(
            "Deprecated compatibility option. Protein atoms are not split into "
            "separate output groups."
        ),
    )
    parser.add_argument(
        "--distance-column",
        help="Distance column name. Default: the column containing 'distance', or the last column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wci_cutoff < args.ci_cutoff:
        raise ValueError("--wci-cutoff must be greater than or equal to --ci-cutoff")

    interactions = build_interactions(
        input_csv=args.input_csv,
        carb_name=args.carb_name,
        drug_column_name=args.drug_column,
        distance_column_name=args.distance_column,
    )
    counts = write_group_csv(
        interactions=interactions,
        output_csv=args.output_csv,
        ci_cutoff=args.ci_cutoff,
        wci_cutoff=args.wci_cutoff,
        ci_label=args.ci_label,
        wci_label=args.wci_label,
        nci_label=args.nci_label,
    )

    print(f"[INFO] Input CSV: {args.input_csv}")
    print(f"[INFO] Output CSV: {args.output_csv}")
    print(f"[INFO] Kept {len(interactions)} {args.carb_name.upper()} carbohydrate residue(s)")
    print(
        "[INFO] Labels: "
        f"{args.ci_label}<={args.ci_cutoff}, "
        f"{args.wci_label}=({args.ci_cutoff}, {args.wci_cutoff}], "
        f"{args.nci_label}>{args.wci_cutoff}"
    )
    print("[INFO] Counts: " + ", ".join(f"{label}={count}" for label, count in counts.items()))


if __name__ == "__main__":
    main()
