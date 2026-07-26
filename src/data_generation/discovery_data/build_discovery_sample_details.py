#!/usr/bin/env python3
"""
Build a CI/NCI discovery sample-detail file from a curated CI seed CSV.

The input file is expected to contain the residues/components that are already
known to be carbohydrate-interacting (CI). For each PDB ID in that file, this
script reads the matching legacy PDB file, collects every occurrence of one
requested residue/component, and writes a new table:

    protein,chain,aa,resnumber,label

Rows from the input CI seed file stay CI. Every other matching residue/component
found in the same PDB file becomes NCI.

Examples
--------
ASN is a standard amino acid, so the default "auto" mode reads ASN only from
ATOM records:

    python build_discovery_sample_details.py \\
        --ci-csv data/discovery/pd1/sample_detailes_pd1_asn_ci.csv \\
        --output-csv data/discovery/pd1/sample_detailes_pd1_asn_ci_nci.csv \\
        --carb ASN

NAG is a carbohydrate/chemical component, so "auto" reads NAG from HETATM
records:

    python build_discovery_sample_details.py \\
        --ci-csv data/discovery/pd1/sample_detailes_pd1_nag_ci.csv \\
        --output-csv data/discovery/pd1/sample_detailes_pd1_nag_ci_nci.csv \\
        --carb NAG

The script intentionally uses only the Python standard library. That makes it
easy to run on LONI or another cluster without managing extra packages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib import error, request


PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
USER_AGENT = "CarbohydratesTSR/1.0"
DEFAULT_OUTPUT_COLUMNS = ["protein", "chain", "aa", "resnumber", "label"]

# The auto record-mode uses this set to decide whether a name is a protein
# residue (ATOM) or a ligand/carbohydrate component (HETATM). ASN is here, NAG is
# not, which matches the CI/NCI workflow requested for the discovery files.
STANDARD_AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass(frozen=True, order=True)
class ResidueKey:
    """A unique residue/component location inside one PDB structure."""

    chain: str
    name: str
    seq_id: str


@dataclass(frozen=True)
class LabeledResidue:
    """A PDB residue/component with the final CI/NCI label assigned."""

    protein: str
    residue: ResidueKey
    label: str


@dataclass
class SeedData:
    """The normalized CI seeds loaded from the source CSV."""

    pdb_ids: List[str]
    ci_by_pdb: Dict[str, Set[ResidueKey]]
    input_rows_read: int
    input_rows_used: int


@dataclass
class WorkerResult:
    """The result from processing one PDB ID."""

    pdb_id: str
    labeled_residues: List[LabeledResidue]
    missing_ci: List[ResidueKey]
    residue_count: int
    pdb_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a discovery sample-detail CSV where curated seed residues are CI "
            "and all other matching residues from the same PDB files are NCI."
        )
    )
    parser.add_argument(
        "--ci-csv",
        required=True,
        type=Path,
        help=(
            "Input CSV containing the known CI rows. The script accepts the current "
            "ASN headers (protein, chain, aa, resnumber, label) and the current NAG "
            "headers (protein, chain, carb, seqnum, lable)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Path for the new CI/NCI sample-detail CSV.",
    )
    parser.add_argument(
        "--carb",
        required=True,
        help=(
            "Residue/component name to collect from each PDB file, for example ASN or NAG. "
            "The name is compared to the PDB residue name."
        ),
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        help=(
            "Optional directory containing existing PDB files. The script checks this "
            "directory first before downloading from RCSB."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/discovery/pdb_cache"),
        help=(
            "Directory used to cache downloaded legacy .pdb files when --pdb-dir does "
            "not already contain the needed structure. Default: data/discovery/pdb_cache."
        ),
    )
    parser.add_argument(
        "--record-mode",
        choices=["auto", "atom", "hetatm", "all"],
        default="auto",
        help=(
            "Which PDB record types to scan. 'auto' uses ATOM for standard amino acids "
            "such as ASN and HETATM for components such as NAG. Default: auto."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only use files already present in --pdb-dir or --download-dir.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the PDB file again even when it already exists in the cache.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of PDB files to download/parse in parallel. Default: 4.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds for each RCSB download. Default: 60.",
    )
    parser.add_argument(
        "--max-pdbs",
        type=int,
        help="Optional debug limit for processing only the first N PDB IDs from the CI file.",
    )
    parser.add_argument(
        "--ci-label",
        default="CI",
        help="Label assigned to rows listed in the source CI CSV. Default: CI.",
    )
    parser.add_argument(
        "--nci-label",
        default="NCI",
        help="Label assigned to all other matching PDB residues/components. Default: NCI.",
    )
    parser.add_argument(
        "--include-missing-ci",
        action="store_true",
        help=(
            "If a seed CI residue is not found in the PDB file, still write it as CI. "
            "Leave this off when the output should be strictly PDB-derived."
        ),
    )
    parser.add_argument(
        "--strict-ci",
        action="store_true",
        help=(
            "Fail the run if any seed CI residue is not found in the matching PDB file. "
            "This is useful when you want to catch numbering or chain mismatches early."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Keep writing output for the PDB files that worked even if one structure "
            "cannot be downloaded or parsed. By default, any PDB processing error fails "
            "the run so incomplete output is not written accidentally."
        ),
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help=(
            "Write one output row per residue/component instead of joining residue "
            "numbers with semicolons by protein/chain/label."
        ),
    )
    return parser


def clean_value(value: object) -> str:
    """Return a stripped CSV/PDB value without turning None into 'None'."""

    if value is None:
        return ""
    return str(value).strip()


def normalize_pdb_id(raw_value: object) -> str:
    """Normalize PDB IDs to upper case because RCSB download URLs use that form."""

    value = clean_value(raw_value).strip('"').strip("'")
    if not value:
        return ""
    return value.split()[0].upper()


def normalize_component_name(raw_value: object) -> str:
    """Normalize residue/component names to the three-letter PDB style."""

    return clean_value(raw_value).upper()


def normalize_seq_id(raw_value: object) -> str:
    """
    Normalize residue sequence IDs without losing insertion codes.

    The source CSVs are strings, but this helper also protects against values
    that may have passed through a spreadsheet and become "58.0".
    """

    value = clean_value(raw_value)
    if re.fullmatch(r"-?\d+\.0", value):
        value = value[:-2]
    return value


def split_seq_ids(raw_value: object) -> List[str]:
    """Split fields such as '49;58;116' into individual residue IDs."""

    value = normalize_seq_id(raw_value)
    if not value:
        return []
    return [part for part in (normalize_seq_id(piece) for piece in re.split(r"[;,]", value)) if part]


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Return unique values while preserving the order in the source file."""

    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def resolve_column(
    fieldnames: Sequence[str],
    role_name: str,
    candidates: Sequence[str],
    required: bool,
) -> Optional[str]:
    """
    Find a CSV column by exact/case-insensitive candidate names.

    The discovery inputs have small schema differences and one historical typo
    ("lable"). Resolving columns here keeps the rest of the code clean.
    """

    lower_to_original = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
        resolved = lower_to_original.get(candidate.lower())
        if resolved is not None:
            return resolved

    if required:
        raise ValueError(
            f"Could not resolve the required {role_name} column. "
            f"Tried: {', '.join(candidates)}. Available columns: {', '.join(fieldnames)}"
        )
    return None


def read_ci_seed_csv(ci_csv: Path, carb: str, ci_label: str) -> SeedData:
    """Load the curated CI rows and normalize them to ResidueKey objects."""

    if not ci_csv.is_file():
        raise FileNotFoundError(f"CI seed CSV does not exist: {ci_csv}")

    with ci_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CI seed CSV is empty: {ci_csv}")

        fieldnames = list(reader.fieldnames)
        protein_column = resolve_column(
            fieldnames,
            role_name="protein/PDB ID",
            candidates=["protein", "pdb", "pdb_id", "entry_id", "identifier"],
            required=True,
        )
        chain_column = resolve_column(
            fieldnames,
            role_name="chain",
            candidates=["chain", "chain_id"],
            required=True,
        )
        name_column = resolve_column(
            fieldnames,
            role_name="residue/component name",
            candidates=["aa", "carb", "carb_name", "residue", "residue_name", "ligand", "entity_name", "name"],
            required=False,
        )
        seq_column = resolve_column(
            fieldnames,
            role_name="residue number",
            candidates=["resnumber", "seqnum", "carb_id", "residue_number", "residue_id", "seq_id"],
            required=True,
        )
        label_column = resolve_column(
            fieldnames,
            role_name="label",
            candidates=["label", "lable", "class", "group"],
            required=False,
        )

        pdb_ids: List[str] = []
        ci_by_pdb: Dict[str, Set[ResidueKey]] = defaultdict(set)
        input_rows_read = 0
        input_rows_used = 0

        for row in reader:
            input_rows_read += 1
            pdb_id = normalize_pdb_id(row.get(protein_column))
            chain = clean_value(row.get(chain_column))
            residue_name = normalize_component_name(row.get(name_column)) if name_column else carb
            label = clean_value(row.get(label_column)).upper() if label_column else ci_label.upper()

            # The source file is a CI seed file. If a mixed file is ever passed
            # here, only the rows explicitly labeled CI should seed the CI set.
            if not pdb_id or residue_name != carb or (label and label != ci_label.upper()):
                continue

            seq_ids = split_seq_ids(row.get(seq_column))
            if not seq_ids:
                print(
                    f"[WARN] Skipping source row {input_rows_read}: missing residue number.",
                    file=sys.stderr,
                )
                continue

            pdb_ids.append(pdb_id)
            for seq_id in seq_ids:
                ci_by_pdb[pdb_id].add(ResidueKey(chain=chain, name=carb, seq_id=seq_id))
            input_rows_used += 1

    pdb_ids = dedupe_preserve_order(pdb_ids)
    if not pdb_ids:
        raise ValueError(
            f"No CI seed rows for carb={carb} were found in {ci_csv}. "
            "Check --carb and the input residue/component column."
        )

    return SeedData(
        pdb_ids=pdb_ids,
        ci_by_pdb=ci_by_pdb,
        input_rows_read=input_rows_read,
        input_rows_used=input_rows_used,
    )


def seq_sort_key(seq_id: str) -> Tuple[int, int, str]:
    """
    Sort residue IDs naturally when possible.

    Numeric IDs sort by number, insertion-code IDs such as 476A stay next to
    their base number, and unusual values fall back to lexical sorting.
    """

    value = normalize_seq_id(seq_id)
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", value)
    if match:
        return (0, int(match.group(1)), match.group(2))
    return (1, 0, value)


def chain_sort_key(chain: str) -> Tuple[int, str]:
    """Sort blank chain IDs last while keeping chain IDs case-sensitive."""

    if chain == "":
        return (1, "")
    return (0, chain)


def record_names_for_mode(carb: str, record_mode: str) -> Set[str]:
    """Translate the requested record mode into PDB record names."""

    if record_mode == "atom":
        return {"ATOM"}
    if record_mode == "hetatm":
        return {"HETATM"}
    if record_mode == "all":
        return {"ATOM", "HETATM"}
    if carb in STANDARD_AMINO_ACIDS:
        return {"ATOM"}
    return {"HETATM"}


def open_text(path: Path) -> Iterator[str]:
    """Open plain-text or gzipped PDB files using a common iterator interface."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def residue_from_pdb_atom_line(line: str, carb: str) -> Optional[ResidueKey]:
    """
    Parse the residue identity from an ATOM/HETATM line.

    Legacy PDB files are fixed-width. The slices below follow the official PDB
    columns: residue name 18-20, chain ID 22, residue sequence 23-26, insertion
    code 27. Python slices are zero-based, so the numbers look one smaller.
    """

    padded_line = line.rstrip("\n").ljust(80)
    residue_name = padded_line[17:20].strip().upper()
    if residue_name != carb:
        return None

    chain = padded_line[21:22].strip()
    seq_number = padded_line[22:26].strip()
    insertion_code = padded_line[26:27].strip()
    if not seq_number:
        return None

    seq_id = f"{seq_number}{insertion_code}" if insertion_code else seq_number
    return ResidueKey(chain=chain, name=residue_name, seq_id=seq_id)


def parse_matching_residues_from_pdb(path: Path, carb: str, record_mode: str) -> Set[ResidueKey]:
    """
    Read one PDB file and return all unique matching residues/components.

    Multiple atom lines for the same residue collapse to one ResidueKey. For
    NMR-style files with MODEL/ENDMDL sections, only the first model is used so
    the same residue is not counted repeatedly across models.
    """

    selected_records = record_names_for_mode(carb, record_mode)
    residues: Set[ResidueKey] = set()

    for line in open_text(path):
        record = line[:6].strip()

        if record == "MODEL":
            model_number = line[10:14].strip()
            if model_number and model_number != "1":
                break
            continue
        if record == "ENDMDL":
            break
        if record not in selected_records:
            continue

        residue = residue_from_pdb_atom_line(line, carb)
        if residue is not None:
            residues.add(residue)

    return residues


def local_pdb_candidates(pdb_id: str, directory: Path) -> List[Path]:
    """Return likely local filenames for a legacy PDB entry."""

    lower_id = pdb_id.lower()
    upper_id = pdb_id.upper()
    names = [
        f"{upper_id}.pdb",
        f"{lower_id}.pdb",
        f"pdb{lower_id}.ent",
        f"{upper_id}.pdb.gz",
        f"{lower_id}.pdb.gz",
        f"pdb{lower_id}.ent.gz",
    ]
    return [directory / name for name in names]


def find_existing_pdb(pdb_id: str, directories: Sequence[Optional[Path]]) -> Optional[Path]:
    """Find a PDB file in the user-provided directory or download cache."""

    for directory in directories:
        if directory is None:
            continue
        for candidate in local_pdb_candidates(pdb_id, directory):
            if candidate.is_file():
                return candidate
    return None


def download_pdb(pdb_id: str, download_dir: Path, timeout: int, force_download: bool) -> Path:
    """Download a legacy PDB file from RCSB and return the cached path."""

    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / f"{pdb_id}.pdb"
    if output_path.is_file() and not force_download:
        return output_path

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    req = request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            output_path.write_bytes(response.read())
    except error.HTTPError as exc:
        if exc.code == 400:
            raise RuntimeError(
                f"Legacy PDB format is not available for {pdb_id}. "
                "RCSB may provide this structure only as mmCIF."
            ) from exc
        raise
    return output_path


def resolve_pdb_path(
    pdb_id: str,
    pdb_dir: Optional[Path],
    download_dir: Path,
    no_download: bool,
    timeout: int,
    force_download: bool,
) -> Path:
    """Use a local PDB file when possible, otherwise download it from RCSB."""

    if not force_download:
        existing_path = find_existing_pdb(pdb_id, directories=[pdb_dir, download_dir])
        if existing_path is not None:
            return existing_path

    if no_download:
        searched_dirs = ", ".join(str(path) for path in [pdb_dir, download_dir] if path is not None)
        raise FileNotFoundError(
            f"No local PDB file found for {pdb_id}. Searched: {searched_dirs or '(none)'}"
        )

    return download_pdb(
        pdb_id=pdb_id,
        download_dir=download_dir,
        timeout=timeout,
        force_download=force_download,
    )


def process_one_pdb(
    pdb_id: str,
    ci_residues: Set[ResidueKey],
    carb: str,
    args: argparse.Namespace,
) -> WorkerResult:
    """Download/read one PDB file, collect matching residues, and assign labels."""

    pdb_path = resolve_pdb_path(
        pdb_id=pdb_id,
        pdb_dir=args.pdb_dir,
        download_dir=args.download_dir,
        no_download=args.no_download,
        timeout=args.timeout,
        force_download=args.force_download,
    )
    pdb_residues = parse_matching_residues_from_pdb(
        path=pdb_path,
        carb=carb,
        record_mode=args.record_mode,
    )

    missing_ci = sorted(ci_residues - pdb_residues, key=lambda item: (chain_sort_key(item.chain), seq_sort_key(item.seq_id)))
    labeled_residues: List[LabeledResidue] = []
    for residue in sorted(pdb_residues, key=lambda item: (chain_sort_key(item.chain), seq_sort_key(item.seq_id))):
        label = args.ci_label if residue in ci_residues else args.nci_label
        labeled_residues.append(LabeledResidue(protein=pdb_id, residue=residue, label=label))

    if args.include_missing_ci:
        for residue in missing_ci:
            labeled_residues.append(LabeledResidue(protein=pdb_id, residue=residue, label=args.ci_label))

    return WorkerResult(
        pdb_id=pdb_id,
        labeled_residues=labeled_residues,
        missing_ci=missing_ci,
        residue_count=len(pdb_residues),
        pdb_path=pdb_path,
    )


def aggregate_labeled_residues(labeled_residues: Iterable[LabeledResidue]) -> List[Dict[str, str]]:
    """Join residue numbers by protein, chain, residue/component name, and label."""

    grouped: Dict[Tuple[str, str, str, str], Set[str]] = defaultdict(set)
    for item in labeled_residues:
        grouped[(item.protein, item.residue.chain, item.residue.name, item.label)].add(item.residue.seq_id)

    rows: List[Dict[str, str]] = []
    for protein, chain, name, label in sorted(
        grouped,
        key=lambda item: (item[0], 0 if item[3].upper() == "CI" else 1, chain_sort_key(item[1]), item[2]),
    ):
        seq_ids = sorted(grouped[(protein, chain, name, label)], key=seq_sort_key)
        rows.append(
            {
                "protein": protein,
                "chain": chain,
                "aa": name,
                "resnumber": ";".join(seq_ids),
                "label": label,
            }
        )
    return rows


def unaggregated_labeled_residues(labeled_residues: Iterable[LabeledResidue]) -> List[Dict[str, str]]:
    """Write one row for each residue/component instead of semicolon grouping."""

    rows: List[Dict[str, str]] = []
    for item in sorted(
        labeled_residues,
        key=lambda value: (
            value.protein,
            0 if value.label.upper() == "CI" else 1,
            chain_sort_key(value.residue.chain),
            seq_sort_key(value.residue.seq_id),
            value.residue.name,
        ),
    ):
        rows.append(
            {
                "protein": item.protein,
                "chain": item.residue.chain,
                "aa": item.residue.name,
                "resnumber": item.residue.seq_id,
                "label": item.label,
            }
        )
    return rows


def write_output_csv(output_csv: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Write the final sample-detail table."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in DEFAULT_OUTPUT_COLUMNS})


def print_missing_ci_warnings(results: Sequence[WorkerResult], max_examples: int = 25) -> int:
    """Print concise warnings for seed CI residues that were not found in PDB ATOM/HETATM records."""

    missing_total = 0
    examples_printed = 0
    for result in sorted(results, key=lambda item: item.pdb_id):
        if not result.missing_ci:
            continue

        missing_total += len(result.missing_ci)
        missing_text = "; ".join(
            f"{residue.chain}:{residue.name}:{residue.seq_id}" for residue in result.missing_ci
        )
        if examples_printed < max_examples:
            print(
                f"[WARN] {result.pdb_id}: {len(result.missing_ci)} CI seed residue(s) "
                f"were not found in {result.pdb_path}: {missing_text}",
                file=sys.stderr,
            )
            examples_printed += 1

    if missing_total and examples_printed >= max_examples:
        print(
            f"[WARN] Suppressed additional missing-CI warnings after {max_examples} PDB entries.",
            file=sys.stderr,
        )
    return missing_total


def main() -> int:
    args = build_parser().parse_args()
    carb = normalize_component_name(args.carb)

    if not carb:
        print("[ERROR] --carb cannot be empty.", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("[ERROR] --workers must be at least 1.", file=sys.stderr)
        return 1

    try:
        seed_data = read_ci_seed_csv(args.ci_csv, carb=carb, ci_label=args.ci_label)
    except Exception as exc:  # noqa: BLE001 - this is a top-level CLI boundary.
        print(f"[ERROR] Failed to read CI seed CSV: {exc}", file=sys.stderr)
        return 1

    pdb_ids = seed_data.pdb_ids[: args.max_pdbs] if args.max_pdbs else seed_data.pdb_ids
    selected_records = ", ".join(sorted(record_names_for_mode(carb, args.record_mode)))

    print(f"[INFO] CI seed CSV: {args.ci_csv}", file=sys.stderr)
    print(f"[INFO] Output CSV: {args.output_csv}", file=sys.stderr)
    print(f"[INFO] carb={carb}; record-mode={args.record_mode} -> scanning {selected_records}", file=sys.stderr)
    print(
        f"[INFO] Read {seed_data.input_rows_read} source row(s), used "
        f"{seed_data.input_rows_used} CI row(s), processing {len(pdb_ids)} PDB ID(s).",
        file=sys.stderr,
    )

    results: List[WorkerResult] = []
    errors: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_pdb,
                pdb_id,
                seed_data.ci_by_pdb.get(pdb_id, set()),
                carb,
                args,
            ): pdb_id
            for pdb_id in pdb_ids
        }

        for future in as_completed(futures):
            pdb_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - collect all worker failures for a useful summary.
                errors.append((pdb_id, str(exc)))
                print(f"[ERROR] {pdb_id}: {exc}", file=sys.stderr)
                continue

            print(
                f"[INFO] {pdb_id}: found {result.residue_count} {carb} residue/component(s) in {result.pdb_path}",
                file=sys.stderr,
            )
            results.append(result)

    if errors and not args.continue_on_error:
        print(
            f"[ERROR] {len(errors)} PDB file(s) failed. Output was not written. "
            "Use --continue-on-error only if partial output is acceptable.",
            file=sys.stderr,
        )
        return 1

    if not results:
        print("[ERROR] No PDB files were processed successfully.", file=sys.stderr)
        return 1

    missing_ci_total = print_missing_ci_warnings(results)
    if missing_ci_total and args.strict_ci:
        print(
            f"[ERROR] {missing_ci_total} CI seed residue(s) were missing from parsed PDB records. "
            "Output was not written because --strict-ci is enabled.",
            file=sys.stderr,
        )
        return 1

    all_labeled_residues = [
        item
        for result in sorted(results, key=lambda value: value.pdb_id)
        for item in result.labeled_residues
    ]
    output_rows = (
        unaggregated_labeled_residues(all_labeled_residues)
        if args.no_aggregate
        else aggregate_labeled_residues(all_labeled_residues)
    )
    write_output_csv(args.output_csv, output_rows)

    label_counts = Counter(item.label for item in all_labeled_residues)
    print(f"[INFO] Wrote {len(output_rows)} row(s) to {args.output_csv}", file=sys.stderr)
    for label in sorted(label_counts, key=lambda value: (0 if value.upper() == "CI" else 1, value)):
        print(f"[INFO] {label}: {label_counts[label]} residue/component location(s)", file=sys.stderr)

    if errors:
        print(f"[WARN] Completed with {len(errors)} skipped PDB file(s).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
