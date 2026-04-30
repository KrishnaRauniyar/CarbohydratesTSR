#!/usr/bin/env python3
"""
Append ligand short codes to the Protein column of a key-summary CSV and,
optionally, rename matching key files in a folder.

The CSV is expected to contain one row per protein-chain and a comma-delimited
ligand list in the order that should be represented in the final name.

Examples:
    CSV row:
        Protein = 10MU_C
        ligands = NAG,NAG

    Generated name:
        10MU_C_NN

    CSV row:
        Protein = 10OP_Y
        ligands = NAG,NAG,BMA,MAN,MAN

    Generated name:
        10OP_Y_NNBMM

    Matching key files:
        10MU_C.keys_theta29_dist18      -> 10MU_C_NN.keys_theta29_dist18
        10MU_C.keys_Freq_theta29_dist18 -> 10MU_C_NN.keys_Freq_theta29_dist18

Usage:
    # Update only the CSV.
    python add_ligands_to_chain.py \
        -i data/glycoprotein_samples/proteinNumKeysDist.csv \
        -o data/glycoprotein_samples/proteinNumKeysDist_with_ligands.csv

    # Preview matching file renames without changing files.
    python add_ligands_to_chain.py \
        -i data/glycoprotein_samples/proteinNumKeysDist.csv \
        --keys_folder path/to/key/files \
        --dry_run

    # Rename matching key files and also write an updated CSV.
    python add_ligands_to_chain.py \
        -i data/glycoprotein_samples/proteinNumKeysDist.csv \
        -o data/glycoprotein_samples/proteinNumKeysDist_with_ligands.csv \
        --keys_folder path/to/key/files
"""

import argparse
import csv
import os
from pathlib import Path


DEFAULT_LIGAND_CODES = {
    "NAG": "N",
    "MAN": "M",
    "BMA": "B",
    "FUC": "F",
    "BGC": "G",
    "GAL": "A",
    "FUL": "U",
    "NDG": "D",
}


def parse_ligands(value):
    """Return ligand names from a comma-delimited CSV cell."""
    if value is None:
        return []
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def build_ligand_suffix(ligand_names, ligand_codes, row_number):
    """Convert ligand names to their configured one-letter suffix."""
    suffix_parts = []
    for ligand_name in ligand_names:
        if ligand_name not in ligand_codes:
            raise ValueError(
                f"Unknown ligand '{ligand_name}' on CSV row {row_number}. "
                "Add it to DEFAULT_LIGAND_CODES in this script."
            )
        suffix_parts.append(ligand_codes[ligand_name])
    return "".join(suffix_parts)


def build_protein_rename_map(
    input_csv,
    protein_column="Protein",
    ligands_column="ligands",
    ligand_codes=None,
):
    """
    Read the CSV and return a mapping from original Protein to labelled Protein.

    For example:
        {
            "10MU_C": "10MU_C_NN",
            "10OP_Y": "10OP_Y_NNBMM",
        }

    This map is the single source of truth for both:
        1. rewriting the Protein column in the output CSV
        2. renaming files in --keys_folder

    Duplicate Protein values are allowed only when they produce the same final
    labelled name. If the same Protein appears with different ligand order/code
    results, the script raises an error because file renaming would be ambiguous.
    """
    ligand_codes = ligand_codes or DEFAULT_LIGAND_CODES
    rename_map = {}

    with open(input_csv, "r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV '{input_csv}' does not contain a header row.")

        missing_columns = [
            column
            for column in (protein_column, ligands_column)
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Input CSV '{input_csv}' is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            protein = str(row.get(protein_column, "")).strip()
            if not protein:
                continue

            ligand_names = parse_ligands(row.get(ligands_column))
            suffix = build_ligand_suffix(
                ligand_names=ligand_names,
                ligand_codes=ligand_codes,
                row_number=row_number,
            )
            labelled_protein = f"{protein}_{suffix}" if suffix else protein

            previous_label = rename_map.get(protein)
            if previous_label is not None and previous_label != labelled_protein:
                raise ValueError(
                    f"Protein '{protein}' appears more than once with different "
                    f"labelled names: '{previous_label}' and '{labelled_protein}'."
                )
            rename_map[protein] = labelled_protein

    return rename_map


def add_ligand_codes_to_csv(
    input_csv,
    output_csv,
    protein_column="Protein",
    ligands_column="ligands",
    rename_map=None,
):
    """Write a copy of input_csv with Protein updated to Protein_suffix."""
    if rename_map is None:
        rename_map = build_protein_rename_map(
            input_csv=input_csv,
            protein_column=protein_column,
            ligands_column=ligands_column,
        )

    with open(input_csv, "r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV '{input_csv}' does not contain a header row.")

        missing_columns = [
            column
            for column in (protein_column, ligands_column)
            if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Input CSV '{input_csv}' is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

        output_dir = os.path.dirname(os.path.abspath(output_csv))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
            writer.writeheader()

            rows_written = 0
            for row in reader:
                protein = str(row.get(protein_column, "")).strip()
                if protein in rename_map:
                    row[protein_column] = rename_map[protein]
                writer.writerow(row)
                rows_written += 1

    return rows_written


def find_labelled_file_name(file_name, rename_map):
    """
    Return the renamed file name when file_name starts with a known Protein id.

    The key files are named like:
        10MU_C.keys_theta29_dist18
        10MU_C.keys_Freq_theta29_dist18

    Because the Protein id is the prefix before the first dot, this function
    replaces only that prefix and preserves the rest of the filename exactly:
        10MU_C.keys_theta29_dist18 -> 10MU_C_NN.keys_theta29_dist18

    Protein ids are sorted longest-first so a longer id is preferred if two ids
    could theoretically share the same prefix.
    """
    for protein in sorted(rename_map, key=len, reverse=True):
        if file_name == protein or file_name.startswith(f"{protein}."):
            labelled_protein = rename_map[protein]
            return f"{labelled_protein}{file_name[len(protein):]}"
    return None


def rename_matching_key_files(keys_folder, rename_map, recursive=False, dry_run=False):
    """
    Rename all files in keys_folder whose names start with a Protein from CSV.

    Args:
        keys_folder: Folder that contains files like 10MU_C.keys_theta29_dist18.
        rename_map: Mapping returned by build_protein_rename_map().
        recursive: If True, process files in subfolders too.
        dry_run: If True, print what would happen but do not rename files.

    Returns:
        A list of (old_path, new_path) tuples for files that were or would be
        renamed.
    """
    folder = Path(keys_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Keys folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Keys folder is not a directory: {folder}")

    paths = folder.rglob("*") if recursive else folder.iterdir()
    planned_renames = []

    for path in paths:
        if not path.is_file():
            continue

        new_name = find_labelled_file_name(path.name, rename_map)
        if new_name is None or new_name == path.name:
            continue

        new_path = path.with_name(new_name)
        if new_path.exists():
            raise FileExistsError(
                f"Cannot rename '{path}' to '{new_path}' because the target "
                "file already exists."
            )

        planned_renames.append((path, new_path))

    for old_path, new_path in planned_renames:
        print(f"[DRY-RUN] {old_path.name} -> {new_path.name}" if dry_run else f"[RENAME] {old_path.name} -> {new_path.name}")
        if not dry_run:
            old_path.rename(new_path)

    return planned_renames


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Append ligand short-code suffixes to the Protein column and/or "
            "rename matching key files in a folder."
        )
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        required=True,
        help="Input CSV containing Protein and ligands columns.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        help=(
            "Output CSV path. Optional when you only want to rename files with "
            "--keys_folder."
        ),
    )
    parser.add_argument(
        "--protein_column",
        default="Protein",
        help="Column containing the protein-chain identifier.",
    )
    parser.add_argument(
        "--ligands_column",
        default="ligands",
        help="Column containing comma-delimited ligand names.",
    )
    parser.add_argument(
        "--keys_folder",
        help=(
            "Folder containing key files to rename, for example files named "
            "10MU_C.keys_theta29_dist18 and 10MU_C.keys_Freq_theta29_dist18."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Rename matching files in subfolders of --keys_folder too.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview key-file renames without changing filenames.",
    )
    args = parser.parse_args()

    if not args.output_csv and not args.keys_folder:
        parser.error("Provide --output_csv, --keys_folder, or both.")

    rename_map = build_protein_rename_map(
        input_csv=args.input_csv,
        protein_column=args.protein_column,
        ligands_column=args.ligands_column,
    )
    print(f"[INFO] Built {len(rename_map)} protein rename mapping(s).")

    if args.output_csv:
        rows_written = add_ligand_codes_to_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            protein_column=args.protein_column,
            ligands_column=args.ligands_column,
            rename_map=rename_map,
        )
        print(f"[INFO] Wrote {rows_written} row(s) to {args.output_csv}")

    if args.keys_folder:
        renamed_files = rename_matching_key_files(
            keys_folder=args.keys_folder,
            rename_map=rename_map,
            recursive=args.recursive,
            dry_run=args.dry_run,
        )
        action = "would be renamed" if args.dry_run else "renamed"
        print(f"[INFO] {len(renamed_files)} file(s) {action}.")


if __name__ == "__main__":
    main()
