"""
Annotate a generalised fingerprint CSV with trailing residue labels.

Why this script exists
----------------------
The repository currently contains a "generalised.csv" file where each row starts
with an identifier such as:

    2V7A_B_391_ASP;0.000,0.558,0.634,...

For downstream DNN work we want the identifier to include the interaction label
at the end, producing labels such as I/NI or CI/NCI:

    2V7A_B_391_ASP_NI;0.000,0.558,0.634,...
    5WT9_G_58_ASN_CI;0,0,2,...

The label is not guessed from the fingerprint itself. Instead, it is looked up
from a residue-level CSV such as:

    data/processed/interaction_min_dist_aminoacid/interaction_residue_min_distance_ASP_balanced.csv

That lookup CSV contains columns:

    protein,chain,residue,seqnum,min_distance,label

This script matches rows using the shared residue identity:

    <protein>_<chain>_<seqnum>_<residue>

and appends the matching label to the identifier.

Design goals
------------
1. Keep the output format as close as possible to the input format.
2. Fail loudly when required labels are missing, instead of silently producing
   partially incorrect data.
3. Be explicit and well-commented so future maintenance is easy.
"""

import argparse
import csv
import re
from pathlib import Path


def normalize_text(value):
    """
    Convert any input value into a clean comparison-friendly string.

    CSV fields can sometimes contain surrounding spaces. Normalizing all
    matching components to stripped strings keeps the join logic predictable.
    """

    if value is None:
        return ""
    return str(value).strip()


def normalize_seqnum(value):
    """
    Normalize residue numbers for matching.

    Treat 58 and 58.0 as the same residue, while preserving insertion-code
    values such as 476A.
    """

    cleaned_value = normalize_text(value)
    if re.fullmatch(r"-?\d+\.0", cleaned_value):
        return cleaned_value[:-2]
    return cleaned_value


def split_seqnums(value):
    """
    Split a sequence-number field into individual residue numbers.

    Newer discovery label files group residues with semicolons, for example
    "33;49;66". Older residue-label files usually contain one seqnum per row.
    Supporting both here keeps the label step independent of how the label file
    was generated.
    """

    cleaned_value = normalize_seqnum(value)
    if not cleaned_value:
        return []
    return [
        normalize_seqnum(part)
        for part in re.split(r"[;,]", cleaned_value)
        if normalize_seqnum(part)
    ]


def build_lookup_key(protein, chain, seqnum, residue):
    """
    Build the common residue identifier shared by both files.

    Expected final format:
        <protein>_<chain>_<seqnum>_<residue>

    Example:
        2V7A_B_391_ASP
    """

    return "_".join(
        [
            normalize_text(protein),
            normalize_text(chain),
            normalize_seqnum(seqnum),
            normalize_text(residue),
        ]
    )


def parse_generalised_identifier(identifier):
    """
    Parse the leftmost identifier from the generalised CSV row.

    The expected structure is:
        <protein>_<chain>_<seqnum>_<residue>

    Example:
        2V7A_B_391_ASP

    We parse from the right so the function is resilient if a future protein
    field ever contains an underscore unexpectedly.
    """

    cleaned_identifier = normalize_text(identifier)
    parts = cleaned_identifier.rsplit("_", 3)

    if len(parts) != 4:
        raise ValueError(
            "Unexpected generalised identifier format. "
            f"Expected '<protein>_<chain>_<seqnum>_<residue>', got: {identifier}"
        )

    protein, chain, seqnum, residue = parts
    return protein, chain, seqnum, residue


def build_label_lookup(label_csv_path):
    """
    Read the residue-label CSV and convert it into a dictionary lookup.

    Output dictionary shape:
        {
            "2V7A_B_391_ASP": "NI",
            "1HPO_B_25_ASP": "I",
            ...
        }

    The script validates required columns up front so failures are immediate and
    descriptive instead of surfacing later as cryptic key errors.
    """

    with label_csv_path.open("r", encoding="utf-8-sig", newline="") as label_file:
        reader = csv.DictReader(label_file)
        if reader.fieldnames is None:
            raise ValueError(f"Label CSV is empty: {label_csv_path}")

        protein_column = resolve_column(reader.fieldnames, "protein", ["protein", "pdb", "pdb_id"])
        chain_column = resolve_column(reader.fieldnames, "chain", ["chain", "chain_id"])
        residue_column = resolve_column(
            reader.fieldnames,
            "residue",
            ["residue", "aa", "carb", "carb_name", "residue_name"],
        )
        seqnum_column = resolve_column(
            reader.fieldnames,
            "seqnum",
            ["seqnum", "resnumber", "carb_id", "residue_number", "residue_id"],
        )
        label_column = resolve_column(reader.fieldnames, "label", ["label", "lable", "class"])

        lookup = {}
        duplicate_keys = set()

        for row in reader:
            label = normalize_text(row[label_column])
            for seqnum in split_seqnums(row[seqnum_column]):
                key = build_lookup_key(
                    protein=row[protein_column],
                    chain=row[chain_column],
                    seqnum=seqnum,
                    residue=row[residue_column],
                )

                if key in lookup and lookup[key] != label:
                    duplicate_keys.add(key)

                lookup[key] = label

        if duplicate_keys:
            duplicate_preview = ", ".join(sorted(list(duplicate_keys))[:10])
            raise ValueError(
                "Conflicting labels were found for the same residue key in the label CSV. "
                f"Example keys: {duplicate_preview}"
            )

        return lookup


def resolve_column(columns, role_name, candidates):
    """
    Resolve a required label CSV column using common historical names.

    The old interaction files use residue/seqnum. The new discovery files use
    aa/resnumber. Both describe the same residue identity, so the lookup builder
    accepts either spelling.
    """

    column_names = list(columns)
    lower_to_original = {column.lower(): column for column in column_names}
    for candidate in candidates:
        if candidate in column_names:
            return candidate
        resolved = lower_to_original.get(candidate.lower())
        if resolved is not None:
            return resolved

    raise ValueError(
        f"Missing required {role_name} column in label CSV. "
        f"Tried: {', '.join(candidates)}. Available columns: {', '.join(column_names)}"
    )


def add_label_to_identifier(identifier, label_lookup, missing_labels, line_number):
    protein, chain, seqnum, residue = parse_generalised_identifier(identifier)
    lookup_key = build_lookup_key(protein, chain, seqnum, residue)

    if lookup_key not in label_lookup:
        missing_labels.append((line_number, lookup_key))
        return None

    label = label_lookup[lookup_key]
    return f"{identifier}_{label}"


def is_header_csv_line(line):
    row = next(csv.reader([line]))
    return bool(row) and normalize_text(row[0]).lower() == "protein name"


def detect_input_format(input_path):
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        for raw_line in infile:
            line = raw_line.strip()

            if not line:
                continue

            if is_header_csv_line(line):
                return "header_csv"

            if ";" in line:
                return "semicolon"

            return "csv"

    return "empty"


def annotate_semicolon_file(input_path, label_lookup, output_path):
    processed_rows = 0
    missing_labels = []

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.rstrip("\n")

            # Skip blank lines quietly so odd whitespace does not break the run.
            if not line.strip():
                continue

            if ";" not in line:
                raise ValueError(
                    f"Line {line_number} in {input_path} does not contain ';' "
                    "between the identifier and the feature vector."
                )

            identifier, feature_payload = line.split(";", 1)
            labeled_identifier = add_label_to_identifier(
                identifier=identifier,
                label_lookup=label_lookup,
                missing_labels=missing_labels,
                line_number=line_number,
            )

            if labeled_identifier is None:
                continue

            outfile.write(f"{labeled_identifier};{feature_payload}\n")
            processed_rows += 1

    return processed_rows, missing_labels


def annotate_csv_file(input_path, label_lookup, output_path):
    processed_rows = 0
    missing_labels = []

    with input_path.open("r", encoding="utf-8", newline="") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, lineterminator="\n")

        for line_number, row in enumerate(reader, start=1):
            if not row or not any(field.strip() for field in row):
                continue

            if normalize_text(row[0]).lower() == "protein name":
                writer.writerow(row)
                continue

            identifier = row[0]
            labeled_identifier = add_label_to_identifier(
                identifier=identifier,
                label_lookup=label_lookup,
                missing_labels=missing_labels,
                line_number=line_number,
            )

            if labeled_identifier is None:
                continue

            row[0] = labeled_identifier
            writer.writerow(row)
            processed_rows += 1

    return processed_rows, missing_labels


def annotate_generalised_file(input_path, label_lookup, output_path):
    """
    Create a labeled copy of the generalised CSV.

    Supported input row formats:
        <identifier>;<comma-separated-values>
        Protein Name,<key-1>,<key-2>,...
        <identifier>,<value-1>,<value-2>,...

    Output row formats:
        <identifier>_<label>;<comma-separated-values>
        Protein Name,<key-1>,<key-2>,...
        <identifier>_<label>,<value-1>,<value-2>,...

    This function works line-by-line instead of loading the full file into a
    pandas DataFrame because the file is large and already stored in a simple
    streaming-friendly text format.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_format = detect_input_format(input_path)

    if input_format == "empty":
        return 0

    if input_format == "semicolon":
        processed_rows, missing_labels = annotate_semicolon_file(
            input_path=input_path,
            label_lookup=label_lookup,
            output_path=output_path,
        )
    else:
        processed_rows, missing_labels = annotate_csv_file(
            input_path=input_path,
            label_lookup=label_lookup,
            output_path=output_path,
        )

    if missing_labels:
        preview = ", ".join(
            [f"line {line_number}: {key}" for line_number, key in missing_labels[:10]]
        )
        raise ValueError(
            "Some generalised identifiers could not be matched to the label CSV. "
            f"Missing examples: {preview}. "
            f"Total unmatched rows: {len(missing_labels)}"
        )

    return processed_rows


def build_default_output_path(input_path):
    """
    Generate a predictable output filename beside the input file.

    Example:
        generalised.csv -> generalised_labeled.csv
    """

    return input_path.with_name(f"{input_path.stem}_labeled{input_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Append I/NI labels to the identifier field of a semicolon-delimited "
            "or header CSV fingerprint file by matching residues against a label CSV."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to the generalised CSV file. "
            "Each row must start with an identifier like 2V7A_B_391_ASP."
        ),
    )
    parser.add_argument(
        "-l",
        "--label-csv",
        type=Path,
        required=True,
        help=(
            "Path to the residue label CSV containing protein, chain, residue, "
            "seqnum, and label columns."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the labeled output file. "
            "If omitted, the script writes '<input_stem>_labeled.csv' beside the input."
        ),
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    label_csv_path = args.label_csv.resolve()
    output_path = args.output.resolve() if args.output else build_default_output_path(input_path)

    print("[INFO] Building residue label lookup")
    label_lookup = build_label_lookup(label_csv_path)
    print(f"[INFO] Loaded {len(label_lookup)} labeled residues from {label_csv_path}")

    print("[INFO] Annotating generalised file")
    processed_rows = annotate_generalised_file(
        input_path=input_path,
        label_lookup=label_lookup,
        output_path=output_path,
    )

    print(f"[INFO] Successfully wrote {processed_rows} labeled rows")
    print(f"[INFO] Output file: {output_path}")


if __name__ == "__main__":
    main()
