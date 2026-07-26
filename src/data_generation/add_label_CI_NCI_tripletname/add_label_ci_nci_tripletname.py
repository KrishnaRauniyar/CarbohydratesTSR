#!/usr/bin/env python3
"""Copy TSR key files and append CI/NCI labels to their filenames."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


KEY_SUFFIXES = {
    "triplets": ".keys_theta29_dist18",
    "frequency": ".keys_Freq_theta29_dist18",
}
IDENTIFIER_ORDERS = (
    "protein-chain-resnumber-name",
    "resnumber-protein-chain-name",
)
VALID_LABELS = {"CI", "NCI"}

COLUMN_ALIASES = {
    "protein": ("protein", "pdb", "pdb_id", "entry_id", "identifier"),
    "chain": ("chain", "chain_id"),
    "residue_name": (
        "aa",
        "carb",
        "carb_name",
        "residue",
        "residue_name",
        "ligand",
        "entity_name",
        "name",
    ),
    "residue_number": (
        "resnumber",
        "carb_res",
        "seqnum",
        "seq_id",
        "residue_number",
        "residue_id",
        "carb_id",
    ),
    "label": ("label", "lable", "group", "class"),
}


@dataclass(frozen=True)
class CopyOperation:
    source: Path
    destination: Path


@dataclass(frozen=True)
class FilenameInterpretation:
    order: str
    residue_key: tuple[str, ...]


def normalize_text(value: str | None) -> str:
    """Normalize residue identity fields for case-insensitive matching."""
    return "" if value is None else str(value).strip().upper()


def normalize_chain(value: str | None) -> str:
    """Normalize whitespace while preserving case-sensitive PDB chain IDs."""
    return "" if value is None else str(value).strip()


def normalize_resnumber(value: str | None) -> str:
    """Treat integer-looking values such as 58 and 58.0 as equivalent."""
    cleaned = normalize_text(value)
    if re.fullmatch(r"-?\d+\.0", cleaned):
        return cleaned[:-2]
    return cleaned


def split_resnumbers(value: str | None) -> list[str]:
    """Expand grouped CSV values such as ``33;49;66`` into residues."""
    cleaned = normalize_text(value)
    if not cleaned:
        return []
    return [
        normalize_resnumber(part)
        for part in re.split(r"[;,]", cleaned)
        if normalize_resnumber(part)
    ]


def build_residue_key(protein: str, chain: str, resnumber: str, aa: str) -> tuple[str, ...]:
    return (
        normalize_text(protein),
        normalize_chain(chain),
        normalize_resnumber(resnumber),
        normalize_text(aa),
    )


def resolve_column(
    fieldnames: list[str],
    role: str,
    explicit_name: str | None = None,
) -> str:
    """Resolve a CSV column from an explicit name or known aliases."""
    columns = {name.strip().lower(): name for name in fieldnames}

    if explicit_name:
        resolved = columns.get(explicit_name.strip().lower())
        if resolved is None:
            raise ValueError(
                f"Requested {role} column '{explicit_name}' was not found. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return resolved

    aliases = COLUMN_ALIASES[role]
    matches = [columns[alias.lower()] for alias in aliases if alias.lower() in columns]
    matches = list(dict.fromkeys(matches))
    if not matches:
        raise ValueError(
            f"Could not identify the {role} column. "
            f"Recognized names: {', '.join(aliases)}. "
            f"Available columns: {', '.join(fieldnames)}. "
            f"Use --{role.replace('_', '-')}-column to specify it explicitly."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple columns could represent {role}: {', '.join(matches)}. "
            f"Use --{role.replace('_', '-')}-column to choose one."
        )
    return matches[0]


def load_label_lookup(
    sample_csv: Path,
    protein_column_name: str | None = None,
    chain_column_name: str | None = None,
    residue_name_column_name: str | None = None,
    residue_number_column_name: str | None = None,
    label_column_name: str | None = None,
) -> dict[tuple[str, ...], str]:
    """Build a residue-to-label lookup from the sample details CSV."""
    if not sample_csv.is_file():
        raise FileNotFoundError(f"Sample CSV does not exist: {sample_csv}")

    with sample_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sample CSV is empty: {sample_csv}")

        protein_column = resolve_column(
            reader.fieldnames, "protein", protein_column_name
        )
        chain_column = resolve_column(reader.fieldnames, "chain", chain_column_name)
        residue_name_column = resolve_column(
            reader.fieldnames, "residue_name", residue_name_column_name
        )
        residue_number_column = resolve_column(
            reader.fieldnames, "residue_number", residue_number_column_name
        )
        label_column = resolve_column(reader.fieldnames, "label", label_column_name)

        print(
            "[INFO] CSV columns: "
            f"protein={protein_column}, chain={chain_column}, "
            f"residue_name={residue_name_column}, "
            f"residue_number={residue_number_column}, label={label_column}"
        )

        lookup: dict[tuple[str, ...], str] = {}
        for line_number, row in enumerate(reader, start=2):
            label = normalize_text(row[label_column])
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Invalid label '{row[label_column]}' on CSV line {line_number}. "
                    f"Expected one of: {', '.join(sorted(VALID_LABELS))}"
                )

            resnumbers = split_resnumbers(row[residue_number_column])
            if not resnumbers:
                raise ValueError(f"Missing resnumber on CSV line {line_number}")

            for resnumber in resnumbers:
                key = build_residue_key(
                    row[protein_column],
                    row[chain_column],
                    resnumber,
                    row[residue_name_column],
                )
                existing_label = lookup.get(key)
                if existing_label is not None and existing_label != label:
                    identifier = "_".join(key)
                    raise ValueError(
                        f"Conflicting labels for {identifier}: "
                        f"{existing_label} and {label}"
                    )
                lookup[key] = label

    if not lookup:
        raise ValueError(f"No labeled residues were found in: {sample_csv}")
    return lookup


def enabled_suffixes(key_file_type: str) -> tuple[str, ...]:
    if key_file_type == "auto":
        return tuple(KEY_SUFFIXES.values())
    return (KEY_SUFFIXES[key_file_type],)


def parse_key_filename(
    file_name: str,
    identifier_order: str,
    suffixes: tuple[str, ...],
) -> tuple[list[FilenameInterpretation], str, str]:
    """Return possible residue interpretations, identifier, and original suffix."""
    suffix = next((value for value in suffixes if file_name.endswith(value)), None)
    if suffix is None:
        raise ValueError(f"Unsupported TSR key filename: {file_name}")

    identifier = file_name[: -len(suffix)]
    parts = identifier.rsplit("_", 3)
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise ValueError(
            "Expected one of the filename formats "
            "'<protein>_<chain>_<resnumber>_<residue-name><suffix>' or "
            "'<resnumber>_<protein>_<chain>_<residue-name><suffix>', "
            f"got: {file_name}"
        )

    first, second, third, residue_name = parts
    interpretations: list[FilenameInterpretation] = []
    requested_orders = (
        IDENTIFIER_ORDERS if identifier_order == "auto" else (identifier_order,)
    )
    for order in requested_orders:
        if order == "protein-chain-resnumber-name":
            protein, chain, resnumber = first, second, third
        else:
            resnumber, protein, chain = first, second, third
        interpretation = FilenameInterpretation(
            order=order,
            residue_key=build_residue_key(protein, chain, resnumber, residue_name),
        )
        if interpretation.residue_key not in {
            item.residue_key for item in interpretations
        }:
            interpretations.append(interpretation)

    return interpretations, identifier, suffix


def collect_copy_operations(
    input_folder: Path,
    output_folder: Path,
    label_lookup: dict[tuple[str, ...], str],
    recursive: bool,
    identifier_order: str = "auto",
    key_file_type: str = "auto",
) -> tuple[list[CopyOperation], list[str]]:
    """Prepare labeled copies and return any files without matching CSV labels."""
    if not input_folder.exists():
        raise FileNotFoundError(f"Input protein folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input protein path is not a folder: {input_folder}")
    if input_folder == output_folder:
        raise ValueError("Input protein folder and output folder must be different")

    if recursive and output_folder.is_relative_to(input_folder):
        raise ValueError(
            "With --recursive, the output folder cannot be inside the input folder"
        )

    suffixes = enabled_suffixes(key_file_type)
    source_paths: set[Path] = set()
    for suffix in suffixes:
        candidates = (
            input_folder.rglob(f"*{suffix}")
            if recursive
            else input_folder.glob(f"*{suffix}")
        )
        source_paths.update(path for path in candidates if path.is_file())
    sorted_source_paths = sorted(source_paths)
    if not sorted_source_paths:
        patterns = ", ".join(f"'*{suffix}'" for suffix in suffixes)
        raise ValueError(
            f"No supported TSR key files ({patterns}) were found in: {input_folder}"
        )

    operations: list[CopyOperation] = []
    unmatched: list[str] = []
    destinations: set[Path] = set()

    for source_path in sorted_source_paths:
        interpretations, identifier, suffix = parse_key_filename(
            source_path.name,
            identifier_order=identifier_order,
            suffixes=suffixes,
        )
        matches = [
            (interpretation, label_lookup[interpretation.residue_key])
            for interpretation in interpretations
            if interpretation.residue_key in label_lookup
        ]
        if not matches:
            attempted = "; ".join(
                f"{item.order}={'_'.join(item.residue_key)}"
                for item in interpretations
            )
            unmatched.append(f"{source_path.name} (tried: {attempted})")
            continue
        if len(matches) > 1:
            matched_text = "; ".join(
                f"{item.order}={'_'.join(item.residue_key)} -> {label}"
                for item, label in matches
            )
            raise ValueError(
                f"Ambiguous identifier order for {source_path.name}: {matched_text}. "
                "Set --identifier-order explicitly."
            )

        _, label = matches[0]

        relative_parent = (
            source_path.parent.relative_to(input_folder) if recursive else Path()
        )
        labeled_name = f"{identifier}_{label}{suffix}"
        destination = output_folder / relative_parent / labeled_name
        if destination in destinations:
            raise ValueError(f"Multiple input files map to the same output: {destination}")
        destinations.add(destination)
        operations.append(CopyOperation(source_path, destination))

    return operations, unmatched


def copy_labeled_files(
    operations: list[CopyOperation], overwrite: bool, dry_run: bool
) -> tuple[int, int]:
    copied = 0
    skipped = 0

    for operation in operations:
        if operation.destination.exists() and not overwrite:
            print(f"[SKIP] Exists: {operation.destination}")
            skipped += 1
            continue

        action = "[DRY-RUN]" if dry_run else "[COPY]"
        print(f"{action} {operation.source} -> {operation.destination}")
        if not dry_run:
            operation.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(operation.source, operation.destination)
        copied += 1

    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match TSR key filenames to a sample CSV and copy them with CI or NCI "
            "appended. Common CSV column aliases and two identifier orders are supported."
        )
    )
    parser.add_argument(
        "-s",
        "--sample-csv",
        "--input-sample-csv",
        required=True,
        type=Path,
        help="CSV containing protein, chain, residue name/number, and label columns.",
    )
    parser.add_argument(
        "-i",
        "--input-protein-folder",
        required=True,
        type=Path,
        help="Folder containing TSR triplet and/or frequency key files.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        required=True,
        type=Path,
        help="Folder where labeled copies will be written.",
    )
    parser.add_argument(
        "--identifier-order",
        choices=["auto", *IDENTIFIER_ORDERS],
        default="auto",
        help=(
            "Filename field order. 'auto' tests both supported orders against the CSV. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--key-file-type",
        choices=["auto", *KEY_SUFFIXES],
        default="auto",
        help=(
            "Process triplet files, frequency files, or both in auto mode. Default: auto."
        ),
    )
    parser.add_argument(
        "--protein-column",
        help="Explicit protein/PDB column name; otherwise recognized aliases are used.",
    )
    parser.add_argument(
        "--chain-column",
        help="Explicit chain column name; otherwise recognized aliases are used.",
    )
    parser.add_argument(
        "--residue-name-column",
        help="Explicit residue/ligand name column; otherwise aliases are used.",
    )
    parser.add_argument(
        "--residue-number-column",
        help="Explicit residue number column; otherwise aliases are used.",
    )
    parser.add_argument(
        "--label-column",
        help="Explicit CI/NCI label column; otherwise recognized aliases are used.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders and preserve their relative paths in the output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite labeled output files that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned copies without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_csv = args.sample_csv.resolve()
    input_folder = args.input_protein_folder.resolve()
    output_folder = args.output_folder.resolve()

    print(f"[INFO] Loading labels from: {sample_csv}")
    label_lookup = load_label_lookup(
        sample_csv,
        protein_column_name=args.protein_column,
        chain_column_name=args.chain_column,
        residue_name_column_name=args.residue_name_column,
        residue_number_column_name=args.residue_number_column,
        label_column_name=args.label_column,
    )
    print(f"[INFO] Loaded {len(label_lookup)} labeled residue(s)")

    operations, unmatched = collect_copy_operations(
        input_folder=input_folder,
        output_folder=output_folder,
        label_lookup=label_lookup,
        recursive=args.recursive,
        identifier_order=args.identifier_order,
        key_file_type=args.key_file_type,
    )
    print(f"[INFO] Matched {len(operations)} TSR key file(s)")

    copied, skipped = copy_labeled_files(
        operations=operations,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    verb = "Would copy" if args.dry_run else "Copied"
    print(f"[INFO] {verb} {copied} file(s); skipped {skipped} existing file(s)")
    print(f"[INFO] Output folder: {output_folder}")

    if unmatched:
        preview = "\n".join(f"  - {item}" for item in unmatched[:10])
        remaining = len(unmatched) - 10
        if remaining > 0:
            preview += f"\n  - ... and {remaining} more"
        print(
            f"[ERROR] Could not find CSV labels for {len(unmatched)} TSR key file(s).\n"
            f"Matched files were still processed. Unmatched examples:\n{preview}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
