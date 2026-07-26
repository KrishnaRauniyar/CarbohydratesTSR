"""
Extract the first column from a local feature-vector CSV.

Feature-vector files in this project commonly store rows like:

    5WT9_G_102_ASN_NCI;0.000,0.821,0.809,...

For downstream steps that only need the row identifier, this script writes a
one-column CSV containing just the leftmost field.
"""

import argparse
import csv
from pathlib import Path


def infer_delimiter(input_path):
    """
    Infer the delimiter from the first non-empty line.

    The generalised feature-vector format uses a semicolon between the
    identifier and a comma-separated vector, so prefer ';' when it appears
    before any comma on the line.
    """
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            semicolon_index = stripped_line.find(";")
            comma_index = stripped_line.find(",")
            if semicolon_index != -1 and (
                comma_index == -1 or semicolon_index < comma_index
            ):
                return ";"
            if comma_index != -1:
                return ","
            if semicolon_index != -1:
                return ";"
            return ","

    return ","


def extract_first_column(
    input_path,
    output_path,
    delimiter,
    output_header,
    has_header=False,
    include_output_header=True,
):
    """Read input_path and write only column 1 to output_path."""
    resolved_delimiter = infer_delimiter(input_path) if delimiter == "auto" else delimiter
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file, delimiter=resolved_delimiter)

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            if include_output_header:
                writer.writerow([output_header])

            for row_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if row_number == 1 and has_header:
                    continue

                writer.writerow([row[0].strip()])
                rows_written += 1

    return rows_written, resolved_delimiter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the first column from a local feature-vector CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input local feature-vector CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output one-column CSV path.",
    )
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="Input delimiter. Use 'auto' to infer, or pass values like ';' or ','.",
    )
    parser.add_argument(
        "--output-header",
        default="identifier",
        help="Header name for the extracted first column.",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Skip the first input row before extracting values.",
    )
    parser.add_argument(
        "--no-output-header",
        action="store_true",
        help="Write values only, without an output header row.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows_written, delimiter = extract_first_column(
        input_path=args.input,
        output_path=args.output,
        delimiter=args.delimiter,
        output_header=args.output_header,
        has_header=args.has_header,
        include_output_header=not args.no_output_header,
    )
    print(
        f"Wrote {rows_written} row(s) from first column to {args.output} "
        f"using delimiter {delimiter!r}."
    )


if __name__ == "__main__":
    main()
