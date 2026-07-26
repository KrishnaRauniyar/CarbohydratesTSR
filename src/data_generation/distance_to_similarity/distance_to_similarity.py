#!/usr/bin/env python3
"""Convert a symmetric normalized-distance matrix to unique similarities.

The expected input format has no header. Each row contains an identifier,
followed by a semicolon and a comma-separated distance vector::

    5WT9_G_102_ASN_NCI;0.000,0.821,0.809,...

The row identifiers also define the column order. Only the upper triangle is
written, excluding the diagonal, so every pair appears exactly once. A
normalized distance is converted to a percentage similarity with::

    similarity = 100 - (distance * 100)
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the upper triangle of a symmetric normalized-distance "
            "matrix to a unique-pair similarity CSV."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Headerless input matrix in identifier;distance,distance,... format.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV with carb_1, carb_2, and similarity columns.",
    )
    return parser.parse_args()


def read_distance_matrix(input_path: Path) -> tuple[list[str], list[list[Decimal]]]:
    """Read and validate the basic structure of a distance matrix."""
    identifiers: list[str] = []
    matrix: list[list[Decimal]] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if ";" not in line:
                raise ValueError(
                    f"Line {line_number} does not contain the required ';' separator."
                )

            identifier, raw_vector = line.split(";", maxsplit=1)
            identifier = identifier.strip()
            if not identifier:
                raise ValueError(f"Line {line_number} has an empty identifier.")

            values: list[Decimal] = []
            for column_number, raw_value in enumerate(raw_vector.split(","), start=1):
                try:
                    value = Decimal(raw_value.strip())
                except InvalidOperation as exc:
                    raise ValueError(
                        f"Invalid distance at line {line_number}, vector column "
                        f"{column_number}: {raw_value!r}."
                    ) from exc

                if not ZERO <= value <= ONE:
                    raise ValueError(
                        f"Distance at line {line_number}, vector column "
                        f"{column_number} is outside [0, 1]: {value}."
                    )
                values.append(value)

            identifiers.append(identifier)
            matrix.append(values)

    if not identifiers:
        raise ValueError("The input matrix is empty.")

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("The input matrix contains duplicate identifiers.")

    matrix_size = len(identifiers)
    for row_number, values in enumerate(matrix, start=1):
        if len(values) != matrix_size:
            raise ValueError(
                f"Matrix is not square: row {row_number} has {len(values)} values, "
                f"but the matrix has {matrix_size} rows."
            )

    return identifiers, matrix


def validate_symmetric(matrix: list[list[Decimal]]) -> None:
    """Require a zero diagonal and matching upper/lower triangles."""
    for row_index, row in enumerate(matrix):
        if row[row_index] != ZERO:
            raise ValueError(
                f"Diagonal value at row {row_index + 1} is not zero: "
                f"{row[row_index]}."
            )

        for column_index in range(row_index + 1, len(matrix)):
            if row[column_index] != matrix[column_index][row_index]:
                raise ValueError(
                    "Matrix is not symmetric at rows/columns "
                    f"{row_index + 1} and {column_index + 1}: "
                    f"{row[column_index]} != {matrix[column_index][row_index]}."
                )


def format_similarity(distance: Decimal) -> str:
    """Return the exact percentage similarity without redundant trailing zeros."""
    similarity = HUNDRED - (distance * HUNDRED)
    formatted = format(similarity, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def write_upper_triangle(
    output_path: Path,
    identifiers: list[str],
    matrix: list[list[Decimal]],
) -> int:
    """Write each non-diagonal pair once using the matrix's upper triangle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_written = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["carb_1", "carb_2", "similarity"])

        for row_index, carb_1 in enumerate(identifiers):
            for column_index in range(row_index + 1, len(identifiers)):
                writer.writerow(
                    [
                        carb_1,
                        identifiers[column_index],
                        format_similarity(matrix[row_index][column_index]),
                    ]
                )
                pairs_written += 1

    return pairs_written


def main() -> None:
    args = parse_args()
    identifiers, matrix = read_distance_matrix(args.input)
    validate_symmetric(matrix)
    pairs_written = write_upper_triangle(args.output, identifiers, matrix)

    print(
        f"Wrote {pairs_written} unique pair(s) from a "
        f"{len(identifiers)} x {len(identifiers)} matrix to {args.output}."
    )


if __name__ == "__main__":
    main()
