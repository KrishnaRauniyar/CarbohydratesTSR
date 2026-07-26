#!/usr/bin/env python3
"""
Build or read a sample-detail file and search TSR key files.

The sample-detail file keeps the identifier from the first column of the
generalised file. Identifiers may be written in either of these forms:

    5WT9_A_102_NAG
    102_5WT9_A_NAG
    7WG3_M_NNBMMM

The first two forms are normalized to the same PDB/chain/sequence/residue
identity. The third form is a whole-chain identifier with a chain group suffix
and no sequence number. Matching TSR key rows are written when the three atom
columns match the requested atoms, independent of atom order.

Two key-row layouts are supported:

    key atom seq atom seq atom seq theta ... coords
    key atom seq residue chain atom seq residue chain atom seq residue chain theta ... coords
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_GENERALIZED_CSV = Path("data/discovery/search/generalised.csv")
DEFAULT_SAMPLE_DETAIL_CSV = Path("data/discovery/search/sample_detail_sep_group.csv")
DEFAULT_OUTPUT_FILE = Path("data/discovery/search/search_tsr_key_matches.csv")
DEFAULT_KEY_SUFFIX = ".keys_theta29_dist18"
ATOM_SOURCE_SUFFIXES = ("_P", "_D")
GROUP_LABELS = {"CI", "NCI", "WCI", "I", "NI"}
IdentifierKey = tuple[str, ...]

KEY_MODE_AUTO = "auto"
KEY_MODE_TSR_KEYS_CHAIN = "tsr_keys_chain"
KEY_MODE_TSR_KEYS_NO_CHAIN = "tsr_keys_no_chain"
KEY_MODE_TSR_CROSS_KEYS = "tsr_cross_keys"
KEY_MODE_CHOICES = (
    KEY_MODE_AUTO,
    KEY_MODE_TSR_KEYS_CHAIN,
    KEY_MODE_TSR_KEYS_NO_CHAIN,
    KEY_MODE_TSR_CROSS_KEYS,
)
KEY_MODE_ALLOWED_IDENTIFIER_TYPES = {
    KEY_MODE_AUTO: {
        "PDB_CHAIN",
        "PDB_CHAIN_GROUP",
        "PDB_CHAIN_SEQ_RESIDUE",
    },
    KEY_MODE_TSR_KEYS_CHAIN: {
        "PDB_CHAIN",
        "PDB_CHAIN_GROUP",
    },
    KEY_MODE_TSR_KEYS_NO_CHAIN: {
        "PDB_CHAIN_SEQ_RESIDUE",
    },
    # Cross-key filenames have varied in this project; the cross mode fixes the
    # row layout while accepting the supported filename identifier styles.
    KEY_MODE_TSR_CROSS_KEYS: {
        "PDB_CHAIN",
        "PDB_CHAIN_GROUP",
        "PDB_CHAIN_SEQ_RESIDUE",
    },
}
KEY_MODE_DEFAULT_LAYOUT = {
    KEY_MODE_AUTO: "auto",
    KEY_MODE_TSR_KEYS_CHAIN: "standard",
    KEY_MODE_TSR_KEYS_NO_CHAIN: "standard",
    KEY_MODE_TSR_CROSS_KEYS: "cross",
}

OUTPUT_KEY_HEADER = [
    "key",
    "atom_1",
    "seq_1",
    "residue_1",
    "chain_1",
    "atom_2",
    "seq_2",
    "residue_2",
    "chain_2",
    "atom_3",
    "seq_3",
    "residue_3",
    "chain_3",
    "theta",
    "theta_degree",
    "distance_bin",
    "max_distance",
    "x_1",
    "y_1",
    "z_1",
    "x_2",
    "y_2",
    "z_2",
    "x_3",
    "y_3",
    "z_3",
]


@dataclass(frozen=True)
class KeyRowLayout:
    name: str
    minimum_fields: int
    atom_indices: tuple[int, int, int]


STANDARD_KEY_LAYOUT = KeyRowLayout(
    name="standard",
    minimum_fields=20,
    atom_indices=(1, 3, 5),
)
CROSS_KEY_LAYOUT = KeyRowLayout(
    name="cross",
    minimum_fields=26,
    atom_indices=(1, 5, 9),
)


@dataclass(frozen=True)
class SampleDetail:
    protein: str
    group: str
    identifier_key: IdentifierKey


@dataclass
class SearchSummary:
    samples: int = 0
    key_files_found: int = 0
    key_files_missing: int = 0
    rows_read: int = 0
    rows_matched: int = 0
    malformed_rows: int = 0
    ambiguous_key_files: int = 0


def normalize_text(value: str | None) -> str:
    return "" if value is None else str(value).strip().upper()


def parse_generalised_first_field(line: str) -> str:
    """Return the first value from a semicolon or comma separated row."""
    if ";" in line:
        return line.split(";", 1)[0].strip()

    row = next(csv.reader([line]))
    return row[0].strip() if row else ""


def looks_like_pdb_id(value: str) -> bool:
    return len(value) == 4 and value.isalnum()


def looks_like_seq_number(value: str) -> bool:
    if not value:
        return False

    core = value[:-1] if value[-1].isalpha() else value
    return core.lstrip("-").isdigit()


def split_optional_group_tokens(tokens: Sequence[str]) -> tuple[list[str], str]:
    """Split a trailing CI/NCI-like label from identifier tokens."""
    if len(tokens) >= 4 and normalize_text(tokens[-1]) in GROUP_LABELS:
        return list(tokens[:-1]), normalize_text(tokens[-1])
    return list(tokens), ""


def split_identifier_group_text(identifier: str) -> tuple[str, str]:
    tokens = [token.strip() for token in identifier.strip().split("_") if token.strip()]
    base_tokens, group = split_optional_group_tokens(tokens)
    return "_".join(base_tokens), group


def identifier_key(identifier: str) -> IdentifierKey:
    """
    Return a canonical key for supported identifier orders.

    Supported:
      * PDB_CHAIN_SEQ_RESIDUE, e.g. 5WT9_A_102_NAG
      * SEQ_PDB_CHAIN_RESIDUE, e.g. 102_5WT9_A_NAG
      * PDB_CHAIN_GROUP, e.g. 7WG3_M_NNBMMM
      * PDB_CHAIN, e.g. 7WG3_M

    A trailing group label such as _CI or _NCI is ignored for the base key.
    PDB IDs and residue/group names are case-normalized, but chain IDs are
    intentionally case-sensitive because PDB chain V and chain v are different.
    """
    cleaned = identifier.strip()
    tokens = [token.strip() for token in cleaned.split("_") if token.strip()]
    tokens, _group = split_optional_group_tokens(tokens)

    if len(tokens) >= 4 and looks_like_pdb_id(tokens[0]) and looks_like_seq_number(tokens[2]):
        return (
            "PDB_CHAIN_SEQ_RESIDUE",
            normalize_text(tokens[0]),
            tokens[1],
            tokens[2],
            normalize_text(tokens[3]),
        )

    if len(tokens) >= 4 and looks_like_seq_number(tokens[0]) and looks_like_pdb_id(tokens[1]):
        return (
            "PDB_CHAIN_SEQ_RESIDUE",
            normalize_text(tokens[1]),
            tokens[2],
            tokens[0],
            normalize_text(tokens[3]),
        )

    if len(tokens) == 3 and looks_like_pdb_id(tokens[0]) and not looks_like_seq_number(tokens[2]):
        return (
            "PDB_CHAIN_GROUP",
            normalize_text(tokens[0]),
            tokens[1],
            normalize_text(tokens[2]),
        )

    if len(tokens) == 2 and looks_like_pdb_id(tokens[0]):
        return ("PDB_CHAIN", normalize_text(tokens[0]), tokens[1])

    return ("RAW", normalize_text(cleaned))


def identifier_key_to_text(key: IdentifierKey) -> str:
    if len(key) == 5 and key[0] == "PDB_CHAIN_SEQ_RESIDUE":
        _, pdb_id, chain, seq_number, residue_name = key
        return f"{pdb_id}_{chain}_{seq_number}_{residue_name}"
    if len(key) == 4 and key[0] == "PDB_CHAIN_GROUP":
        _, pdb_id, chain, chain_group = key
        return f"{pdb_id}_{chain}_{chain_group}"
    if len(key) == 3 and key[0] == "PDB_CHAIN":
        _, pdb_id, chain = key
        return f"{pdb_id}_{chain}"
    return "_".join(key[1:] if key[:1] == ("RAW",) else key)


def key_allowed_for_mode(key: IdentifierKey, key_mode: str) -> bool:
    allowed_types = KEY_MODE_ALLOWED_IDENTIFIER_TYPES[key_mode]
    return key[:1] != ("RAW",) and key[0] in allowed_types


def expected_identifier_message(key_mode: str) -> str:
    if key_mode == KEY_MODE_TSR_KEYS_CHAIN:
        return "'PDB_CHAIN_GROUP' or 'PDB_CHAIN'"
    if key_mode == KEY_MODE_TSR_KEYS_NO_CHAIN:
        return "'PDB_CHAIN_SEQ_RESIDUE' or 'SEQ_PDB_CHAIN_RESIDUE'"
    if key_mode == KEY_MODE_TSR_CROSS_KEYS:
        return (
            "'PDB_CHAIN_SEQ_RESIDUE', 'SEQ_PDB_CHAIN_RESIDUE', "
            "'PDB_CHAIN_GROUP', or 'PDB_CHAIN'"
        )
    return (
        "'PDB_CHAIN_SEQ_RESIDUE', 'SEQ_PDB_CHAIN_RESIDUE', "
        "'PDB_CHAIN_GROUP', or 'PDB_CHAIN'"
    )


def normalize_explicit_group(value: str) -> str:
    cleaned = str(value).strip()
    normalized = normalize_text(cleaned)
    return normalized if normalized in GROUP_LABELS else cleaned


def split_sample_identifier(
    identifier: str,
    key_mode: str = KEY_MODE_AUTO,
    explicit_group: str = "",
) -> SampleDetail:
    cleaned = identifier.strip()
    protein, embedded_group = split_identifier_group_text(cleaned)
    group = normalize_explicit_group(explicit_group) if explicit_group else embedded_group
    if explicit_group and embedded_group and normalize_text(explicit_group) != embedded_group:
        raise ValueError(
            f"Identifier group '{embedded_group}' conflicts with explicit "
            f"group '{explicit_group}' for: {identifier}"
        )

    key = identifier_key(cleaned)
    if not key_allowed_for_mode(key, key_mode):
        raise ValueError(
            f"Expected identifier format {expected_identifier_message(key_mode)} "
            f"for key mode '{key_mode}', but found: {identifier}"
        )
    return SampleDetail(protein=protein, group=group, identifier_key=key)


def build_sample_detail_file(
    generalised_csv: Path,
    sample_detail_csv: Path,
    key_mode: str = KEY_MODE_AUTO,
) -> list[SampleDetail]:
    """Create sample_detail_sep_group.csv and return the parsed rows."""
    if not generalised_csv.is_file():
        raise FileNotFoundError(f"Generalised CSV does not exist: {generalised_csv}")

    sample_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    samples_by_key: dict[tuple[IdentifierKey, str], SampleDetail] = {}

    with generalised_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            first_field = parse_generalised_first_field(line)
            if not first_field:
                continue

            # A header is not expected, but this keeps the script forgiving.
            if line_number == 1 and normalize_text(first_field) in {
                "PROTEIN",
                "PROTEIN NAME",
            }:
                continue

            sample = split_sample_identifier(first_field, key_mode=key_mode)
            samples_by_key[(sample.identifier_key, sample.group)] = sample

    samples = list(samples_by_key.values())
    if not samples:
        raise ValueError(f"No sample identifiers were found in: {generalised_csv}")

    with sample_detail_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["protein", "group"])
        for sample in samples:
            writer.writerow([sample.protein, sample.group])

    return samples


def normalize_column_name(name: str | None) -> str:
    if name is None:
        return ""
    return "".join(character for character in str(name).strip().lower() if character.isalnum())


def row_value(row: dict[str, str], aliases: Sequence[str]) -> str:
    normalized_aliases = {normalize_column_name(alias) for alias in aliases}
    for column, value in row.items():
        if normalize_column_name(column) in normalized_aliases:
            return "" if value is None else str(value).strip()
    return ""


def split_multi_value(value: str) -> list[str]:
    parts: list[str] = []
    for chunk in str(value).replace(",", ";").split(";"):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def identifiers_from_structured_row(row: dict[str, str], key_mode: str) -> list[str]:
    """Build identifier(s) from common sample-detail column sets."""
    protein = row_value(row, ("protein", "pdb", "pdb_id", "pdbid", "protein_id"))
    chain = row_value(row, ("chain", "chain_id", "chainid", "carb_chain"))
    residue_name = row_value(
        row,
        (
            "carb_name",
            "carb",
            "aa",
            "residue",
            "residue_name",
            "resname",
            "entity_name",
        ),
    )
    seq_values = row_value(
        row,
        (
            "carb_id",
            "carb_res",
            "seqnum",
            "seq_num",
            "resnumber",
            "residue_number",
            "res_seq",
            "seq",
        ),
    )
    chain_group = row_value(
        row,
        (
            "chain_group",
            "chaingroup",
            "group_code",
            "groupcode",
            "ligand_group",
            "ligandgroup",
            "ligands_code",
            "ligandscode",
        ),
    )

    if not protein:
        return []

    if key_mode == KEY_MODE_TSR_KEYS_CHAIN:
        if chain and chain_group:
            return [f"{protein}_{chain}_{chain_group}"]
        if chain:
            return [f"{protein}_{chain}"]
        return []

    if chain and residue_name and seq_values:
        return [
            f"{protein}_{chain}_{seq_value}_{residue_name}"
            for seq_value in split_multi_value(seq_values)
        ]

    if chain and chain_group:
        return [f"{protein}_{chain}_{chain_group}"]

    if chain:
        return [f"{protein}_{chain}"]

    return []


def samples_from_search_row(row: dict[str, str], key_mode: str) -> list[SampleDetail]:
    group = row_value(row, ("group", "label", "class"))
    identifier = row_value(
        row,
        (
            "identifier",
            "protein_identifier",
            "proteinidentifier",
            "protein_name",
            "proteinname",
            "protein",
        ),
    )

    if identifier:
        key = identifier_key(identifier)
        if key_allowed_for_mode(key, key_mode):
            return [
                split_sample_identifier(
                    identifier,
                    key_mode=key_mode,
                    explicit_group=group,
                )
            ]

    samples: list[SampleDetail] = []
    for structured_identifier in identifiers_from_structured_row(row, key_mode):
        samples.append(
            split_sample_identifier(
                structured_identifier,
                key_mode=key_mode,
                explicit_group=group,
            )
        )
    return samples


def first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if line.strip():
                return line
    return ""


def search_input_has_header(path: Path) -> bool:
    first_line = first_nonempty_line(path)
    if not first_line:
        return False

    first_field = parse_generalised_first_field(first_line)
    if normalize_column_name(first_field) in {
        "protein",
        "proteinname",
        "identifier",
        "proteinidentifier",
        "pdb",
        "pdbid",
    }:
        return True

    row = next(csv.reader([first_line.replace(";", ",")]))
    normalized_columns = {normalize_column_name(column) for column in row}
    known_columns = {
        "protein",
        "chain",
        "carbname",
        "carbid",
        "seqnum",
        "resnumber",
        "group",
        "identifier",
    }
    return bool(normalized_columns & known_columns) and identifier_key(first_field)[:1] == ("RAW",)


def read_existing_search_input(search_input_csv: Path, key_mode: str) -> list[SampleDetail]:
    """Read an existing search input/sample-detail CSV and return parsed rows."""
    if not search_input_csv.is_file():
        raise FileNotFoundError(f"Search input CSV does not exist: {search_input_csv}")

    samples_by_key: dict[tuple[IdentifierKey, str], SampleDetail] = {}

    if search_input_has_header(search_input_csv):
        with search_input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Search input CSV has no header: {search_input_csv}")

            for row_number, row in enumerate(reader, start=2):
                if not any(str(value).strip() for value in row.values() if value is not None):
                    continue
                row_samples = samples_from_search_row(row, key_mode)
                if not row_samples:
                    raise ValueError(
                        f"Could not build a supported identifier from row {row_number} "
                        f"in {search_input_csv}"
                    )
                for sample in row_samples:
                    samples_by_key[(sample.identifier_key, sample.group)] = sample
    else:
        with search_input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                first_field = parse_generalised_first_field(line)
                if not first_field:
                    continue
                sample = split_sample_identifier(first_field, key_mode=key_mode)
                samples_by_key[(sample.identifier_key, sample.group)] = sample

    samples = list(samples_by_key.values())
    if not samples:
        raise ValueError(f"No sample identifiers were found in: {search_input_csv}")
    return samples


def parse_atoms(raw_atoms: Sequence[str]) -> list[str]:
    atoms: list[str] = []
    for value in raw_atoms:
        for part in value.replace(",", " ").replace(";", " ").split():
            cleaned = part.strip().strip("\"'")
            if cleaned:
                atoms.append(cleaned)

    if len(atoms) != 3:
        raise ValueError(
            "Exactly three atoms are required, for example: "
            f'--atoms "N, O, P". Parsed {len(atoms)} atom(s): {atoms}'
        )
    return atoms


def strip_atom_source_suffix(atom: str) -> str:
    """Remove known source suffixes used by some cross-key atom labels."""
    for suffix in ATOM_SOURCE_SUFFIXES:
        if atom.endswith(suffix):
            return atom[: -len(suffix)]
    return atom


def normalize_atom_for_match(atom: str, ignore_source_suffix: bool) -> str:
    cleaned = atom.strip()
    if ignore_source_suffix:
        return strip_atom_source_suffix(cleaned)
    return cleaned


def atoms_match(
    row_atoms: Sequence[str],
    requested_atoms: Sequence[str],
    ignore_source_suffix: bool = False,
) -> bool:
    """Compare exact atom labels without caring about their order in the row."""
    cleaned_row_atoms = [
        normalize_atom_for_match(atom, ignore_source_suffix)
        for atom in row_atoms
    ]
    cleaned_requested_atoms = [
        normalize_atom_for_match(atom, ignore_source_suffix)
        for atom in requested_atoms
    ]
    return Counter(cleaned_row_atoms) == Counter(cleaned_requested_atoms)


def collect_key_files(
    key_dir: Path,
    suffix: str,
    recursive: bool,
    key_mode: str,
) -> dict[IdentifierKey, dict[str, Path]]:
    """Map each normalized identifier to its TSR key file."""
    if not key_dir.exists():
        raise FileNotFoundError(f"Key directory does not exist: {key_dir}")
    if not key_dir.is_dir():
        raise NotADirectoryError(f"Key path is not a directory: {key_dir}")

    pattern = f"*{suffix}"
    candidates = key_dir.rglob(pattern) if recursive else key_dir.glob(pattern)

    key_files: dict[IdentifierKey, dict[str, Path]] = {}
    for path in sorted(candidate for candidate in candidates if candidate.is_file()):
        file_identifier = path.name[: -len(suffix)]
        key = identifier_key(file_identifier)
        _base_identifier, group = split_identifier_group_text(file_identifier)
        if not key_allowed_for_mode(key, key_mode):
            print(
                f"[WARN] Skipping key file with unsupported identifier format "
                f"for key mode '{key_mode}': {path}",
                file=sys.stderr,
            )
            continue

        grouped_paths = key_files.setdefault(key, {})
        previous = grouped_paths.get(group)
        if previous is not None:
            raise ValueError(
                "Found more than one key file for "
                f"{identifier_key_to_text_with_group(key, group)}: "
                f"{previous} and {path}"
            )
        grouped_paths[group] = path

    return key_files


def identifier_key_to_text_with_group(key: IdentifierKey, group: str) -> str:
    base_text = identifier_key_to_text(key)
    return f"{base_text}_{group}" if group else base_text


def resolve_key_file(
    sample: SampleDetail,
    key_files: dict[IdentifierKey, dict[str, Path]],
) -> tuple[Path | None, str | None]:
    grouped_paths = key_files.get(sample.identifier_key)
    if not grouped_paths:
        return None, None

    if sample.group and sample.group in grouped_paths:
        return grouped_paths[sample.group], None

    if "" in grouped_paths:
        return grouped_paths[""], None

    if not sample.group and len(grouped_paths) == 1:
        return next(iter(grouped_paths.values())), None

    available_groups = ", ".join(group or "<no group>" for group in sorted(grouped_paths))
    sample_text = identifier_key_to_text_with_group(sample.identifier_key, sample.group)
    return (
        None,
        f"{sample_text} has ambiguous grouped key files: {available_groups}",
    )


def key_layout_from_choice(choice: str) -> KeyRowLayout | None:
    if choice in {"standard", "normal"}:
        return STANDARD_KEY_LAYOUT
    if choice == "cross":
        return CROSS_KEY_LAYOUT
    return None


def resolve_key_layout_choice(key_mode: str, key_layout: str) -> str:
    if key_layout != "auto":
        return key_layout
    return KEY_MODE_DEFAULT_LAYOUT[key_mode]


def detect_key_row_layout(
    fields: Sequence[str],
    key_layout: str,
) -> KeyRowLayout | None:
    selected_layout = key_layout_from_choice(key_layout)
    if selected_layout is not None:
        if len(fields) >= selected_layout.minimum_fields:
            return selected_layout
        return None

    if len(fields) >= CROSS_KEY_LAYOUT.minimum_fields:
        return CROSS_KEY_LAYOUT

    if len(fields) >= STANDARD_KEY_LAYOUT.minimum_fields:
        return STANDARD_KEY_LAYOUT

    return None


def cleaned_field(fields: Sequence[str], index: int) -> str:
    return fields[index].strip()


def normalized_key_row(fields: Sequence[str], layout: KeyRowLayout) -> list[str]:
    if layout == CROSS_KEY_LAYOUT:
        return [
            cleaned_field(fields, index)
            for index in range(CROSS_KEY_LAYOUT.minimum_fields)
        ]

    return [
        cleaned_field(fields, 0),
        cleaned_field(fields, 1),
        cleaned_field(fields, 2),
        "",
        "",
        cleaned_field(fields, 3),
        cleaned_field(fields, 4),
        "",
        "",
        cleaned_field(fields, 5),
        cleaned_field(fields, 6),
        "",
        "",
        cleaned_field(fields, 7),
        cleaned_field(fields, 8),
        cleaned_field(fields, 9),
        cleaned_field(fields, 10),
        cleaned_field(fields, 11),
        cleaned_field(fields, 12),
        cleaned_field(fields, 13),
        cleaned_field(fields, 14),
        cleaned_field(fields, 15),
        cleaned_field(fields, 16),
        cleaned_field(fields, 17),
        cleaned_field(fields, 18),
        cleaned_field(fields, 19),
    ]


def iter_matching_key_rows(
    key_file: Path,
    requested_atoms: Sequence[str],
    summary: SearchSummary,
    ignore_atom_source_suffix: bool,
    key_layout: str,
) -> Iterable[list[str]]:
    with key_file.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            fields = line.rstrip("\n\r").split("\t")
            layout = detect_key_row_layout(fields, key_layout)
            if layout is None:
                summary.malformed_rows += 1
                print(
                    f"[WARN] Skipping row {line_number} in {key_file}: "
                    f"unsupported {len(fields)}-field key layout",
                    file=sys.stderr,
                )
                continue

            summary.rows_read += 1
            row_atoms = [
                fields[index]
                for index in layout.atom_indices
            ]
            if atoms_match(row_atoms, requested_atoms, ignore_atom_source_suffix):
                summary.rows_matched += 1
                yield normalized_key_row(fields, layout)


def write_search_results(
    samples: Sequence[SampleDetail],
    key_files: dict[IdentifierKey, dict[str, Path]],
    output_file: Path,
    requested_atoms: Sequence[str],
    ignore_atom_source_suffix: bool,
    key_layout: str,
    include_group: bool,
    no_header: bool,
    strict_missing_keys: bool,
) -> SearchSummary:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = SearchSummary(samples=len(samples))
    missing: list[str] = []
    ambiguous: list[str] = []

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")

        if not no_header:
            left_columns = ["protein", "group"] if include_group else ["protein"]
            writer.writerow(left_columns + OUTPUT_KEY_HEADER)

        for sample in samples:
            key_file, ambiguity = resolve_key_file(sample, key_files)
            if key_file is None:
                if ambiguity:
                    summary.ambiguous_key_files += 1
                    ambiguous.append(ambiguity)
                else:
                    summary.key_files_missing += 1
                    missing.append(
                        identifier_key_to_text_with_group(
                            sample.identifier_key,
                            sample.group,
                        )
                    )
                continue

            summary.key_files_found += 1
            prefix = (
                [sample.protein, sample.group]
                if include_group
                else [sample.protein]
            )
            for fields in iter_matching_key_rows(
                key_file=key_file,
                requested_atoms=requested_atoms,
                summary=summary,
                ignore_atom_source_suffix=ignore_atom_source_suffix,
                key_layout=key_layout,
            ):
                writer.writerow(prefix + fields)

    if missing:
        preview = ", ".join(missing[:10])
        message = (
            f"Missing key files for {len(missing)} sample(s). "
            f"Examples: {preview}"
        )
        if strict_missing_keys:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}", file=sys.stderr)

    if ambiguous:
        preview = "; ".join(ambiguous[:10])
        message = (
            f"Ambiguous key files for {len(ambiguous)} sample(s). "
            f"Examples: {preview}"
        )
        if strict_missing_keys:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}", file=sys.stderr)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or read a sample-detail CSV and search "
            "matching TSR key files for an unordered atom triplet."
        )
    )
    parser.add_argument(
        "--generalised-csv",
        type=Path,
        help=(
            "Optional generalised CSV. When provided, the script creates "
            "--sample-detail-csv from its first column before searching."
        ),
    )
    parser.add_argument(
        "--search-input-csv",
        type=Path,
        help=(
            "Existing search/sample-detail CSV to read when --generalised-csv "
            "is not provided. If omitted, --sample-detail-csv is read."
        ),
    )
    parser.add_argument(
        "--sample-detail-csv",
        type=Path,
        default=DEFAULT_SAMPLE_DETAIL_CSV,
        help=(
            "Output sample-detail CSV when --generalised-csv is provided, or "
            "the default existing search input when --search-input-csv is "
            f"omitted. Default: {DEFAULT_SAMPLE_DETAIL_CSV}"
        ),
    )
    parser.add_argument(
        "--key-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing files like 5WT9_A_102_NAG.keys_theta29_dist18 "
            "102_5WT9_A_NAG.keys_theta29_dist18, or "
            "7WG3_M_NNBMMM.keys_theta29_dist18."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=(
            "Output CSV with protein name plus matching key rows. "
            f"Default: {DEFAULT_OUTPUT_FILE}"
        ),
    )
    parser.add_argument(
        "--atoms",
        nargs="+",
        required=True,
        help='Three exact atom names to search for, for example: --atoms "N, O, P"',
    )
    parser.add_argument(
        "--ignore-atom-source-suffix",
        action="store_true",
        help=(
            "Treat atom labels ending in _P or _D as their base atom names. "
            "For example, OD1, OD1_P, and OD1_D all compare as OD1."
        ),
    )
    parser.add_argument(
        "--key-mode",
        choices=KEY_MODE_CHOICES,
        default=KEY_MODE_AUTO,
        help=(
            "Key file mode. tsr_keys_chain uses PDB_CHAIN or PDB_CHAIN_GROUP "
            "filenames such as 7WG3_M_NNBMMM and standard key rows. "
            "tsr_keys_no_chain uses residue-specific TSR-key filenames and "
            "standard key rows. tsr_cross_keys uses cross-key rows. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--key-layout",
        choices=("auto", "standard", "normal", "cross"),
        default="auto",
        help=(
            "Optional key row layout override. Use auto for both standard rows "
            "(atom/seq triplets) and cross rows (atom/seq/residue/chain "
            "triplets), or let --key-mode choose the layout. Default: auto."
        ),
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_KEY_SUFFIX,
        help=f"Key file suffix. Default: {DEFAULT_KEY_SUFFIX}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for key files recursively inside --key-dir.",
    )
    parser.add_argument(
        "--include-group",
        action="store_true",
        help="Include the legacy group column in the search output.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Write only data rows in the search output.",
    )
    parser.add_argument(
        "--strict-missing-keys",
        action="store_true",
        help="Fail if any sample has no matching key file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_atoms = parse_atoms(args.atoms)
    resolved_key_layout = resolve_key_layout_choice(
        key_mode=args.key_mode,
        key_layout=args.key_layout,
    )

    if args.generalised_csv is not None:
        samples = build_sample_detail_file(
            generalised_csv=args.generalised_csv,
            sample_detail_csv=args.sample_detail_csv,
            key_mode=args.key_mode,
        )
        search_input_csv = args.sample_detail_csv
        sample_detail_message = f"Sample-detail file created: {args.sample_detail_csv}"
    else:
        search_input_csv = args.search_input_csv or args.sample_detail_csv
        samples = read_existing_search_input(
            search_input_csv=search_input_csv,
            key_mode=args.key_mode,
        )
        sample_detail_message = f"Search input file read: {search_input_csv}"

    key_files = collect_key_files(
        key_dir=args.key_dir,
        suffix=args.suffix,
        recursive=args.recursive,
        key_mode=args.key_mode,
    )
    summary = write_search_results(
        samples=samples,
        key_files=key_files,
        output_file=args.output_file,
        requested_atoms=requested_atoms,
        ignore_atom_source_suffix=args.ignore_atom_source_suffix,
        key_layout=resolved_key_layout,
        include_group=args.include_group,
        no_header=args.no_header,
        strict_missing_keys=args.strict_missing_keys,
    )

    print(f"[INFO] {sample_detail_message}")
    print(f"[INFO] Key mode: {args.key_mode}")
    print(f"[INFO] Key layout: {resolved_key_layout}")
    print(f"[INFO] Search output file: {args.output_file}")
    print(f"[INFO] Samples read: {summary.samples}")
    print(f"[INFO] Key files found: {summary.key_files_found}")
    print(f"[INFO] Key files missing: {summary.key_files_missing}")
    print(f"[INFO] Ambiguous key files: {summary.ambiguous_key_files}")
    print(f"[INFO] Key rows read: {summary.rows_read}")
    print(f"[INFO] Matching rows written: {summary.rows_matched}")
    if summary.malformed_rows:
        print(
            f"[WARN] Malformed key rows skipped: {summary.malformed_rows}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
