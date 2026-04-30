#!/usr/bin/env python3
"""
Generate glycoprotein carbohydrate rows in the format:
protein,chain,carb_name,carb_id,group

The script can either:
1. Query RCSB directly with a search such as:
   full-text "Glycoprotein" AND Glycosylation Site exists
2. Read a file containing PDB IDs that you already exported from RCSB

For each PDB ID it downloads the legacy PDB file, parses HETATM residues,
uses LINK records to keep glycan-like residues attached to the protein,
aggregates duplicate residue numbers, and writes a fresh output CSV.
Merging with an existing CSV is available only when explicitly requested.
"""





from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib import error, request


SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DEFAULT_COLUMNS = ["protein", "chain", "carb_name", "carb_id", "group"]
EXCLUDED_HET_RESIDUES = {"HOH", "WAT", "DOD", "UNX"}
GlycosylationRole = str


@dataclass(frozen=True, order=True)
class ResidueKey:
    chain: str
    residue_name: str
    seq_id: str


@dataclass
class PDBParseResult:
    atom_residues: Set[ResidueKey]
    het_residues: Set[ResidueKey]
    residue_graph: Dict[ResidueKey, Set[ResidueKey]]
    atom_seq_ids_by_chain: Dict[str, Set[str]]
    het_heavy_atom_counts: Dict[ResidueKey, int]


@dataclass
class WorkerResult:
    pdb_id: str
    rows: List[Dict[str, str]]
    selection_mode_used: str


class LegacyPDBFormatUnavailableError(RuntimeError):
    """Raised when RCSB does not provide a legacy .pdb file for an entry."""


def build_legacy_pdb_unavailable_message(pdb_id: str) -> str:
    return (
        f"Legacy PDB format is not available for entry {pdb_id} "
        f"(RCSB .pdb download returned HTTP 400 Bad Request). "
        "RCSB provides some structures only in PDBx/mmCIF format, not in the old legacy PDB format. "
        "Common reasons include multi-character chain IDs, more than 62 chains, more than 99999 atoms, "
        "complex beta-sheet topology, B-factors above 999.99, 5-character chemical component IDs, "
        "or extended PDB IDs."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate glycoprotein carbohydrate rows from RCSB search results "
            "or from a local list of PDB IDs."
        )
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to write the generated CSV.",
    )
    parser.add_argument(
        "--input-csv",
        help=(
            "Optional existing carbohydrate CSV. By default this is used only as a carb_name whitelist. "
            "It is merged into the output only if --merge-input-csv is also provided."
        ),
    )
    parser.add_argument(
        "--merge-input-csv",
        action="store_true",
        help=(
            "Merge rows from --input-csv into the final output. "
            "If not set, the script writes only newly generated rows."
        ),
    )
    parser.add_argument(
        "--pdb-ids-file",
        help=(
            "Optional text/CSV file containing PDB IDs. If omitted, IDs are fetched from RCSB "
            "using the search arguments below."
        ),
    )
    parser.add_argument(
        "--download-dir",
        default="data/glycoprotein_samples/pdb_cache",
        help="Directory used to cache downloaded PDB files.",
    )
    parser.add_argument(
        "--group",
        default="PD1_PDL1",
        help="Group label assigned to all newly generated rows.",
    )
    parser.add_argument(
        "--search-text",
        default="Glycoprotein",
        help=(
            "RCSB full-text search value. This mirrors the generic text box in the advanced search UI. "
            "Ignored when --pdb-ids-file is used."
        ),
    )
    parser.add_argument(
        "--require-glycosylation-site",
        action="store_true",
        help=(
            "Require the RCSB Glycosylation Site field to exist. This maps to the UI filter "
            "'Glycosylation Site' -> 'is not empty'."
        ),
    )
    parser.add_argument(
        "--no-require-glycosylation-site",
        dest="require_glycosylation_site",
        action="store_false",
        help="Disable the Glycosylation Site exists filter.",
    )
    parser.set_defaults(require_glycosylation_site=True)
    parser.add_argument(
        "--glycosylation-types",
        nargs="+",
        choices=[
            "C-Mannosylation",
            "N-Glycosylation",
            "O-Glycosylation",
            "S-Glycosylation",
        ],
        help=(
            "Optional exact glycosylation roles. If provided, these are used instead of a generic "
            "exists filter on the Glycosylation Site field."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Number of RCSB search results to fetch per page.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Optional cap on the number of PDB IDs processed.",
    )
    parser.add_argument(
        "--save-pdb-ids",
        help="Optional file path where the resolved PDB ID list will be written, one per line.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "linked", "all"],
        default="auto",
        help=(
            "'linked' keeps only HETATM residues connected to the protein by LINK records, "
            "'all' keeps all HETATM residues except solvent, and 'auto' uses linked residues "
            "when present then falls back to all HETATM residues."
        ),
    )
    parser.add_argument(
        "--min-heavy-atoms-per-residue",
        type=int,
        default=3,
        help=(
            "Drop HETATM residues that contain fewer than this many heavy atoms. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--exclude-between-atom-residues",
        action="store_true",
        help=(
            "Exclude HETATM residues whose sequence number falls inside the ATOM residue numbering "
            "range for the same chain."
        ),
    )
    parser.add_argument(
        "--allow-between-atom-residues",
        dest="exclude_between_atom_residues",
        action="store_false",
        help="Keep HETATM residues even if they fall inside the ATOM residue numbering range.",
    )
    parser.set_defaults(exclude_between_atom_residues=True)
    parser.add_argument(
        "--known-carb-csv",
        help=(
            "Optional CSV with a carb_name column. Used as a whitelist when you want to restrict "
            "output to known carbohydrate residue names."
        ),
    )
    parser.add_argument(
        "--known-carb-only",
        action="store_true",
        help="Apply the carbohydrate residue whitelist to the final selected residues.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads used for download + parsing.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds for RCSB requests.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download PDB files even if they already exist in the cache directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the PDB ID list and search request, then stop before downloading PDB files.",
    )
    parser.add_argument(
        "--print-search-json",
        action="store_true",
        help="Print the RCSB search JSON payload to stdout before running the search.",
    )
    return parser


def http_post_json(url: str, payload: Dict[str, object], timeout: int) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CarbohydratesTSR/1.0",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body.strip():
            return {}
        return json.loads(body)


def build_search_payload(
    search_text: Optional[str],
    require_glycosylation_site: bool,
    glycosylation_types: Optional[Sequence[GlycosylationRole]],
    start: int,
    rows: int,
) -> Dict[str, object]:
    nodes: List[Dict[str, object]] = []

    if search_text:
        nodes.append(
            {
                "type": "terminal",
                "service": "full_text",
                "parameters": {
                    "value": search_text,
                },
            }
        )

    if glycosylation_types:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_struct_conn.role",
                    "operator": "in",
                    "value": list(glycosylation_types),
                },
            }
        )
    elif require_glycosylation_site:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_struct_conn.role",
                    "operator": "exists",
                },
            }
        )

    if not nodes:
        raise ValueError(
            "At least one search clause is required. Provide --search-text or enable a glycosylation filter."
        )

    query: Dict[str, object]
    if len(nodes) == 1:
        query = nodes[0]
    else:
        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        }

    return {
        "return_type": "entry",
        "query": query,
        "request_options": {
            "paginate": {
                "start": start,
                "rows": rows,
            },
            "results_verbosity": "compact",
        },
    }


def normalize_identifier(raw_value: str) -> Optional[str]:
    value = str(raw_value).strip()
    if not value:
        return None

    value = value.strip('"').strip("'")
    value = value.split()[0]

    if "." in value:
        value = value.split(".", 1)[0]
    if "-" in value and len(value.split("-", 1)[0]) >= 4:
        value = value.split("-", 1)[0]

    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        return None
    return value.upper()


def deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def fetch_pdb_ids_from_search(args: argparse.Namespace) -> List[str]:
    pdb_ids: List[str] = []
    start = 0

    while True:
        payload = build_search_payload(
            search_text=args.search_text,
            require_glycosylation_site=args.require_glycosylation_site,
            glycosylation_types=args.glycosylation_types,
            start=start,
            rows=args.page_size,
        )

        if args.print_search_json and start == 0:
            print(json.dumps(payload, indent=2))

        response = http_post_json(SEARCH_ENDPOINT, payload, timeout=args.timeout)
        result_set = response.get("result_set") or []
        if not result_set:
            break

        page_ids: List[str] = []
        for item in result_set:
            if isinstance(item, str):
                pdb_id = normalize_identifier(item)
            elif isinstance(item, dict):
                pdb_id = normalize_identifier(item.get("identifier", ""))
            else:
                pdb_id = None

            if pdb_id:
                page_ids.append(pdb_id)

        pdb_ids.extend(page_ids)

        if args.max_results and len(pdb_ids) >= args.max_results:
            pdb_ids = pdb_ids[: args.max_results]
            break

        if len(result_set) < args.page_size:
            break

        start += args.page_size

    return deduplicate_preserve_order(pdb_ids)


def load_pdb_ids_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"PDB ID file not found: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []

            field_lookup = {name.lower(): name for name in reader.fieldnames}
            preferred_columns = [
                "protein",
                "pdb_id",
                "entry_id",
                "identifier",
                "pdb",
            ]

            selected_column = None
            for column in preferred_columns:
                if column in field_lookup:
                    selected_column = field_lookup[column]
                    break

            values: List[str] = []
            for row in reader:
                if selected_column:
                    candidate = normalize_identifier(row.get(selected_column, ""))
                    if candidate:
                        values.append(candidate)
                    continue

                for value in row.values():
                    candidate = normalize_identifier(value)
                    if candidate:
                        values.append(candidate)
                        break

            return deduplicate_preserve_order(values)

    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = normalize_identifier(line)
        if candidate:
            values.append(candidate)
    return deduplicate_preserve_order(values)


def save_pdb_ids(pdb_ids: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(pdb_ids) + ("\n" if pdb_ids else ""), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [
            {column: str(row.get(column, "")).strip() for column in reader.fieldnames}
            for row in reader
        ]


def load_known_carb_names(path: Path) -> Set[str]:
    rows = read_csv_rows(path)
    carb_names = {
        row.get("carb_name", "").strip()
        for row in rows
        if row.get("carb_name", "").strip()
    }
    return carb_names


def parse_residue_seq_sort_key(seq_id: str) -> Tuple[int, int, str]:
    match = re.fullmatch(r"(-?\d+)([A-Za-z]?)", seq_id)
    if match:
        return (0, int(match.group(1)), match.group(2))
    return (1, 0, seq_id)


def split_carb_id_field(carb_id_value: str) -> List[str]:
    return [part.strip() for part in str(carb_id_value).split(";") if part.strip()]


def is_heavy_atom_line(line: str) -> bool:
    element = line[76:78].strip().upper()
    if not element:
        atom_name = "".join(character for character in line[12:16] if character.isalpha()).upper()
        if atom_name.startswith(("H", "D")):
            return False
        return True
    return element not in {"H", "D"}


def merge_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str], Set[str]] = defaultdict(set)

    for row in rows:
        protein = str(row.get("protein", "")).strip()
        chain = str(row.get("chain", "")).strip()
        carb_name = str(row.get("carb_name", "")).strip()
        group_name = str(row.get("group", "")).strip()
        carb_ids = split_carb_id_field(str(row.get("carb_id", "")).strip())

        if not (protein and carb_name):
            continue

        grouped[(protein, chain, carb_name, group_name)].update(carb_ids)

    merged_rows = []
    for protein, chain, carb_name, group_name in sorted(grouped.keys()):
        carb_ids = sorted(grouped[(protein, chain, carb_name, group_name)], key=parse_residue_seq_sort_key)
        merged_rows.append(
            {
                "protein": protein,
                "chain": chain,
                "carb_name": carb_name,
                "carb_id": ";".join(carb_ids),
                "group": group_name,
            }
        )
    return merged_rows


def residue_key_from_atom_like_line(line: str) -> ResidueKey:
    residue_name = line[17:20].strip()
    chain = line[21:22].strip()
    seq_number = line[22:26].strip()
    insertion_code = line[26:27].strip()
    seq_id = f"{seq_number}{insertion_code}" if insertion_code else seq_number
    return ResidueKey(chain=chain, residue_name=residue_name, seq_id=seq_id)


def residue_key_from_link_left(line: str) -> ResidueKey:
    residue_name = line[17:20].strip()
    chain = line[21:22].strip()
    seq_number = line[22:26].strip()
    insertion_code = line[26:27].strip()
    seq_id = f"{seq_number}{insertion_code}" if insertion_code else seq_number
    return ResidueKey(chain=chain, residue_name=residue_name, seq_id=seq_id)


def residue_key_from_link_right(line: str) -> ResidueKey:
    residue_name = line[47:50].strip()
    chain = line[51:52].strip()
    seq_number = line[52:56].strip()
    insertion_code = line[56:57].strip()
    seq_id = f"{seq_number}{insertion_code}" if insertion_code else seq_number
    return ResidueKey(chain=chain, residue_name=residue_name, seq_id=seq_id)


def parse_pdb_file(path: Path) -> PDBParseResult:
    atom_residues: Set[ResidueKey] = set()
    het_residues: Set[ResidueKey] = set()
    residue_graph: Dict[ResidueKey, Set[ResidueKey]] = defaultdict(set)
    atom_seq_ids_by_chain: Dict[str, Set[str]] = defaultdict(set)
    het_heavy_atom_counts: Dict[ResidueKey, int] = defaultdict(int)

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()

            if record == "MODEL":
                model_number = line[10:14].strip()
                if model_number and model_number != "1":
                    break
                continue
            if record == "ENDMDL":
                break

            if record == "ATOM":
                residue_key = residue_key_from_atom_like_line(line)
                atom_residues.add(residue_key)
                atom_seq_ids_by_chain[residue_key.chain].add(residue_key.seq_id)
                continue

            if record == "HETATM":
                residue_key = residue_key_from_atom_like_line(line)
                if residue_key.residue_name not in EXCLUDED_HET_RESIDUES:
                    het_residues.add(residue_key)
                    if is_heavy_atom_line(line):
                        het_heavy_atom_counts[residue_key] += 1
                continue

            if record == "LINK":
                left = residue_key_from_link_left(line)
                right = residue_key_from_link_right(line)

                if (
                    left.residue_name in EXCLUDED_HET_RESIDUES
                    or right.residue_name in EXCLUDED_HET_RESIDUES
                ):
                    continue

                residue_graph[left].add(right)
                residue_graph[right].add(left)

    return PDBParseResult(
        atom_residues=atom_residues,
        het_residues=het_residues,
        residue_graph=residue_graph,
        atom_seq_ids_by_chain=atom_seq_ids_by_chain,
        het_heavy_atom_counts=het_heavy_atom_counts,
    )


def select_linked_het_residues(parsed: PDBParseResult) -> Set[ResidueKey]:
    seeds = {
        residue
        for residue in parsed.het_residues
        if any(neighbor in parsed.atom_residues for neighbor in parsed.residue_graph.get(residue, set()))
    }

    selected = set(seeds)
    stack = list(seeds)
    while stack:
        current = stack.pop()
        for neighbor in parsed.residue_graph.get(current, set()):
            if neighbor in parsed.het_residues and neighbor not in selected:
                selected.add(neighbor)
                stack.append(neighbor)
    return selected


def residue_is_between_atom_residues(
    residue: ResidueKey,
    atom_seq_ids_by_chain: Dict[str, Set[str]],
) -> bool:
    chain_seq_ids = atom_seq_ids_by_chain.get(residue.chain)
    if not chain_seq_ids:
        return False

    residue_key = parse_residue_seq_sort_key(residue.seq_id)
    atom_sort_keys = sorted(parse_residue_seq_sort_key(seq_id) for seq_id in chain_seq_ids)
    return atom_sort_keys[0] < residue_key < atom_sort_keys[-1]


def choose_residues(
    parsed: PDBParseResult,
    selection_mode: str,
    known_carb_names: Optional[Set[str]],
    known_carb_only: bool,
    min_heavy_atoms_per_residue: int,
    exclude_between_atom_residues: bool,
) -> Tuple[Set[ResidueKey], str]:
    linked_residues = select_linked_het_residues(parsed)

    if selection_mode == "linked":
        selected = linked_residues
        mode_used = "linked"
    elif selection_mode == "all":
        selected = set(parsed.het_residues)
        mode_used = "all"
    else:
        if linked_residues:
            selected = linked_residues
            mode_used = "linked"
        else:
            selected = set(parsed.het_residues)
            mode_used = "all"

    if known_carb_names and (known_carb_only or mode_used == "all"):
        selected = {
            residue for residue in selected if residue.residue_name in known_carb_names
        }

    if min_heavy_atoms_per_residue > 0:
        selected = {
            residue
            for residue in selected
            if parsed.het_heavy_atom_counts.get(residue, 0) >= min_heavy_atoms_per_residue
        }

    if exclude_between_atom_residues:
        selected = {
            residue
            for residue in selected
            if not residue_is_between_atom_residues(
                residue=residue,
                atom_seq_ids_by_chain=parsed.atom_seq_ids_by_chain,
            )
        }

    return selected, mode_used


def rows_from_residues(
    pdb_id: str,
    residues: Iterable[ResidueKey],
    group_name: str,
) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for residue in residues:
        grouped[(residue.chain, residue.residue_name)].add(residue.seq_id)

    rows = []
    for (chain, residue_name), seq_ids in sorted(grouped.items()):
        rows.append(
            {
                "protein": pdb_id,
                "chain": chain,
                "carb_name": residue_name,
                "carb_id": ";".join(sorted(seq_ids, key=parse_residue_seq_sort_key)),
                "group": group_name,
            }
        )
    return rows


def download_pdb_if_needed(
    pdb_id: str,
    download_dir: Path,
    timeout: int,
    force_download: bool,
) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / f"{pdb_id}.pdb"
    if output_path.exists() and not force_download:
        return output_path

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    req = request.Request(url, headers={"User-Agent": "CarbohydratesTSR/1.0"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            output_path.write_bytes(response.read())
    except error.HTTPError as exc:
        if exc.code == 400:
            raise LegacyPDBFormatUnavailableError(
                build_legacy_pdb_unavailable_message(pdb_id)
            ) from exc
        raise
    return output_path


def process_single_pdb(
    pdb_id: str,
    args: argparse.Namespace,
    known_carb_names: Optional[Set[str]],
) -> WorkerResult:
    pdb_path = download_pdb_if_needed(
        pdb_id=pdb_id,
        download_dir=Path(args.download_dir),
        timeout=args.timeout,
        force_download=args.force_download,
    )
    parsed = parse_pdb_file(pdb_path)
    residues, mode_used = choose_residues(
        parsed=parsed,
        selection_mode=args.selection_mode,
        known_carb_names=known_carb_names,
        known_carb_only=args.known_carb_only,
        min_heavy_atoms_per_residue=args.min_heavy_atoms_per_residue,
        exclude_between_atom_residues=args.exclude_between_atom_residues,
    )
    rows = rows_from_residues(pdb_id=pdb_id, residues=residues, group_name=args.group)
    return WorkerResult(pdb_id=pdb_id, rows=rows, selection_mode_used=mode_used)


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in DEFAULT_COLUMNS})


def resolve_pdb_ids(args: argparse.Namespace) -> List[str]:
    if args.pdb_ids_file:
        pdb_ids = load_pdb_ids_from_file(Path(args.pdb_ids_file))
    else:
        pdb_ids = fetch_pdb_ids_from_search(args)

    if args.max_results:
        pdb_ids = pdb_ids[: args.max_results]
    return pdb_ids


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.pdb_ids_file and not (args.search_text or args.require_glycosylation_site or args.glycosylation_types):
        parser.error("Provide --pdb-ids-file or at least one RCSB search clause.")

    try:
        pdb_ids = resolve_pdb_ids(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to resolve PDB IDs: {exc}", file=sys.stderr)
        return 1

    if not pdb_ids:
        print("No PDB IDs found.", file=sys.stderr)
        return 1

    print(f"Resolved {len(pdb_ids)} PDB IDs.", file=sys.stderr)

    if args.save_pdb_ids:
        save_pdb_ids(pdb_ids, Path(args.save_pdb_ids))

    if args.dry_run:
        print("Dry run complete. No PDB files were downloaded.", file=sys.stderr)
        return 0

    known_carb_names: Optional[Set[str]] = None
    if args.known_carb_csv:
        known_carb_names = load_known_carb_names(Path(args.known_carb_csv))
    elif args.input_csv:
        known_carb_names = load_known_carb_names(Path(args.input_csv))

    generated_rows: List[Dict[str, str]] = []
    linked_count = 0
    fallback_all_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_single_pdb, pdb_id, args, known_carb_names): pdb_id
            for pdb_id in pdb_ids
        }

        for future in as_completed(futures):
            pdb_id = futures[future]
            try:
                result = future.result()
            except LegacyPDBFormatUnavailableError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
                continue
            except error.HTTPError as exc:
                print(
                    f"[WARN] Failed to download {pdb_id}: HTTP {exc.code} {exc.reason}",
                    file=sys.stderr,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to process {pdb_id}: {exc}", file=sys.stderr)
                continue

            if result.selection_mode_used == "linked":
                linked_count += 1
            else:
                fallback_all_count += 1

            generated_rows.extend(result.rows)

    print(
        f"Generated {len(generated_rows)} rows from {linked_count + fallback_all_count} processed structures "
        f"(linked mode: {linked_count}, all-HETATM mode: {fallback_all_count}).",
        file=sys.stderr,
    )

    all_rows: List[Dict[str, str]] = []
    if args.input_csv and args.merge_input_csv:
        all_rows.extend(read_csv_rows(Path(args.input_csv)))
    all_rows.extend(generated_rows)

    merged_rows = merge_rows(all_rows)
    write_csv(Path(args.output_csv), merged_rows)

    print(f"Wrote {len(merged_rows)} rows to {args.output_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
