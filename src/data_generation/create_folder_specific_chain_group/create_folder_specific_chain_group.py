#!/usr/bin/env python3
"""
Copy key files for selected chain groups into a new folder.

The expected key filenames look like:
    10MU_J_NN.keys_theta29_dist18
    10MU_J_NN.keys_Freq_theta29_dist18

In those examples, the chain group is NN. Supplying --groups NN copies both
matching files. Supplying --groups NN N A copies files whose chain group is one
of NN, N, or A.

Examples:
    python create_folder_specific_chain_group.py \
        --input_folder /path/to/key/files \
        --output_folder /path/to/NN_files \
        --groups NN

    python create_folder_specific_chain_group.py \
        --input_folder /path/to/key/files \
        --output_folder /path/to/selected_files \
        --groups NN N A

    python create_folder_specific_chain_group.py \
        --input_folder /path/to/key/files \
        --output_folder /path/to/selected_files \
        --groups "[NN, N, A]" \
        --dry_run
"""

import argparse
import ast
import shutil
from pathlib import Path


def normalize_group(group):
    """Return a clean, case-normalized chain group string."""
    return str(group).strip().strip("'\"").upper()


def parse_groups(values):
    """
    Parse group arguments from either shell-style or list-style input.

    Supported forms:
        --groups NN N A
        --groups NN,N,A
        --groups "[NN, N, A]"
        --groups "['NN', 'N', 'A']"
    """
    if not values:
        raise ValueError("At least one group must be provided.")

    raw_text = " ".join(values).strip()
    parsed_values = []

    if raw_text.startswith("[") and raw_text.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_text)
            if isinstance(parsed, (list, tuple, set)):
                parsed_values = list(parsed)
            else:
                parsed_values = [parsed]
        except (SyntaxError, ValueError):
            parsed_values = raw_text.strip("[]").split(",")
    else:
        for value in values:
            parsed_values.extend(value.split(","))

    groups = {normalize_group(value) for value in parsed_values if normalize_group(value)}
    if not groups:
        raise ValueError("No valid groups were provided.")
    return groups


def extract_chain_group(file_name):
    """
    Extract the chain group from a key filename.

    For 10MU_J_NN.keys_theta29_dist18, this returns NN.
    Files without a .keys marker or without a third underscore-delimited token
    are ignored by returning None.
    """
    stem = file_name.split(".keys", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return parts[-1].upper()


def file_matches_group(path, groups, match_mode):
    """Return True when path should be copied for the selected groups."""
    chain_group = extract_chain_group(path.name)
    if chain_group is None:
        return False

    if match_mode == "exact":
        return chain_group in groups
    if match_mode == "contains":
        return any(group in chain_group for group in groups)

    raise ValueError(f"Unsupported match mode: {match_mode}")


def collect_matching_files(input_folder, groups, recursive=False, match_mode="exact"):
    """Return sorted key files matching the selected chain groups."""
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")

    paths = folder.rglob("*") if recursive else folder.iterdir()
    matches = [
        path
        for path in paths
        if path.is_file() and ".keys" in path.name and file_matches_group(path, groups, match_mode)
    ]
    return sorted(matches)


def copy_matching_files(
    input_folder,
    output_folder,
    groups,
    recursive=False,
    match_mode="exact",
    overwrite=False,
    dry_run=False,
):
    """Copy matching files into output_folder and return copied file count."""
    source_folder = Path(input_folder)
    destination_folder = Path(output_folder)
    matching_files = collect_matching_files(
        input_folder=source_folder,
        groups=groups,
        recursive=recursive,
        match_mode=match_mode,
    )

    if not dry_run:
        destination_folder.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    for source_path in matching_files:
        if recursive:
            relative_path = source_path.relative_to(source_folder)
            destination_path = destination_folder / relative_path
        else:
            destination_path = destination_folder / source_path.name

        if destination_path.exists() and not overwrite:
            print(f"[SKIP] Exists: {destination_path}")
            skipped_count += 1
            continue

        print(f"[DRY-RUN] {source_path} -> {destination_path}" if dry_run else f"[COPY] {source_path} -> {destination_path}")
        if not dry_run:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        copied_count += 1

    return copied_count, skipped_count, len(matching_files)


def main():
    parser = argparse.ArgumentParser(
        description="Copy .keys files for selected chain groups into a new folder."
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        help="Folder containing files like 10MU_J_NN.keys_theta29_dist18.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="Folder where matching files will be copied.",
    )
    parser.add_argument(
        "-g",
        "--groups",
        nargs="+",
        required=True,
        help=(
            "Chain groups to copy. Examples: --groups NN, --groups NN N A, "
            "or --groups '[NN, N, A]'."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input subfolders too. Relative subfolder paths are preserved.",
    )
    parser.add_argument(
        "--match_mode",
        choices=("exact", "contains"),
        default="exact",
        help=(
            "Use exact to match the filename chain-group token. Use contains "
            "for substring matching inside that token."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the output folder when names already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print files that would be copied without creating or copying files.",
    )
    args = parser.parse_args()

    groups = parse_groups(args.groups)
    print(f"[INFO] Selected group(s): {', '.join(sorted(groups))}")
    print(f"[INFO] Input folder: {args.input_folder}")
    print(f"[INFO] Output folder: {args.output_folder}")
    print(f"[INFO] Match mode: {args.match_mode}")
    print(f"[INFO] Recursive: {args.recursive}")
    print(f"[INFO] Dry run: {args.dry_run}")

    copied_count, skipped_count, matched_count = copy_matching_files(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        groups=groups,
        recursive=args.recursive,
        match_mode=args.match_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "would be copied" if args.dry_run else "copied"
    print(f"[INFO] Matched {matched_count} file(s).")
    print(f"[INFO] {copied_count} file(s) {action}; {skipped_count} skipped.")


if __name__ == "__main__":
    main()
