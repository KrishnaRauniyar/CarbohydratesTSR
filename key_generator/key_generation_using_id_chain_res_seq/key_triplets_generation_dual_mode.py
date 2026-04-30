"""
Generate atom-triplet keys from a CSV of target entities.

The mathematical key-generation workflow is intentionally preserved exactly as
it existed before this edit. The changes in this module are limited to:

1. mode-aware CSV column resolution
2. residue-vs-ligand atom selection
3. stronger validation and clearer CLI usage

Supported calculation modes
---------------------------
1. ``all_4_id_chain_res_seqnum``
   Process one explicit entity target per row. Conceptually the four identifier
   roles are:

   * protein
   * chain
   * residue/entity name
   * seqnum

   The script prefers a single name-column role plus ``seqnum``. By default it
   will look for common headers such as ``residue``, ``ligand``, ``name``, or
   ``entity_name`` for the name role, and ``seqnum`` for the sequence-number
   role.

2. ``id_chain``
   Group CSV rows by ``protein`` and ``chain`` only. In this mode the CSV is
   used only to provide unique protein-chain pairs; the atom-selection logic
   then operates on the entire matching chain scope for the selected
   ``entity_type``.

Supported entity types
----------------------
* ``ligand``
  Select heavy atoms from ``HETATM`` records. This preserves the previous
  ligand-oriented behavior.

* ``residue``
  Select non-hydrogen, non-deuterium atoms from matching ``ATOM`` records for
  the requested residue and sequence number. The key calculation itself is
  unchanged; only the atom-selection stage differs.

Example commands
----------------
Residue mode using residue/seqnum columns:
    python src/key_triplets_generator/key_triplets_generation_dual_mode.py \
        -p pdb_files \
        -c data/processed/interaction_min_dist_aminoacid/interaction_residue_min_distance_ASP_balanced.csv \
        -l path/to/lexical.csv \
        -o proteins \
        --mode all_4_id_chain_res_seqnum \
        --entity_type residue

Ligand mode using a generic name column:
    python src/key_triplets_generator/key_triplets_generation_dual_mode.py \
        -p pdb_files \
        -c sample_details_ligands.csv \
        -l path/to/lexical.csv \
        -o proteins \
        --mode all_4_id_chain_res_seqnum \
        --entity_type ligand \
        --entity_name_column ligand

Grouped id_chain mode:
    python src/key_triplets_generator/key_triplets_generation_dual_mode.py \
        -p pdb_files \
        -c sample_details_ligands.csv \
        -l path/to/lexical.csv \
        -o proteins \
        --mode id_chain \
        --entity_type ligand
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys

import pandas as pd
import requests
from joblib import Parallel, delayed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SRC_DIR)
for import_path in (SCRIPT_DIR, SRC_DIR, REPO_ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

try:
    from key_triplets_generator.key_triplets.utils.theta_utils import thetaClass
    from key_triplets_generator.key_triplets.utils.distance_utils import dist12Class
except ModuleNotFoundError:
    from utils.theta_utils import thetaClass
    from utils.distance_utils import dist12Class

CSV_FILE_PATH = ""
CSV_FILE_LEXICAL_PATH = ""
PDB_DIR_PATH = ""
PROTEIN_DIR_PATH = ""
SUMMARY_CSV_PATH = ""
dtheta = 29
dLen = 18
numOfLabels = 112

MODE_ALL_4 = "all_4_id_chain_res_seqnum"
MODE_ID_CHAIN = "id_chain"
ENTITY_LIGAND = "ligand"
ENTITY_RESIDUE = "residue"
DEFAULT_ATOM_COUNT_CONFIG = os.path.join(SCRIPT_DIR, "expected_atom_counts.json")
EXPECTED_LIGAND_ATOM_COUNTS = {}
EXPECTED_RESIDUE_ATOM_COUNTS = {}
SUMMARY_HEADER = [
    "Protein",
    "#atoms",
    "#keys",
    "#keys_with_freq",
    "max_distance",
    "min_distance",
    "ligands",
    "chain",
    "seqnum",
]


def clean_csv_value(value):
    """Return a consistently stripped string for CSV-derived values."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_seqnums(value, row_number=None, column_name="seqnum"):
    """
    Parse a sequence-number column into a list of individual residue identifiers.

    The source file may contain a single sequence number such as ``3`` or a
    semicolon-delimited list such as ``4;5``. Empty fragments are ignored.
    """
    seqnums = []
    raw_value = clean_csv_value(value)
    if not raw_value:
        if row_number is not None:
            print(
                f"Warning: skipping row {row_number} because '{column_name}' is missing."
            )
        return seqnums

    for seq_value in raw_value.split(";"):
        cleaned_value = seq_value.strip()
        if cleaned_value:
            seqnums.append(cleaned_value)

    if not seqnums and row_number is not None:
        print(
            f"Warning: skipping row {row_number} because '{column_name}' value '{raw_value}' "
            "does not contain a valid residue sequence number."
        )

    return seqnums


def normalize_mode(mode):
    """Validate and normalize the requested calculation mode."""
    normalized = clean_csv_value(mode).lower()
    if normalized not in {MODE_ALL_4, MODE_ID_CHAIN}:
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: "
            f"{MODE_ALL_4}, {MODE_ID_CHAIN}."
        )
    return normalized


def normalize_entity_type(entity_type):
    """Validate and normalize the requested atom-selection entity type."""
    normalized = clean_csv_value(entity_type).lower()
    if normalized not in {ENTITY_LIGAND, ENTITY_RESIDUE}:
        raise ValueError(
            f"Unsupported entity type '{entity_type}'. Expected one of: "
            f"{ENTITY_LIGAND}, {ENTITY_RESIDUE}."
        )
    return normalized


def build_identifier_for_all4(protein, chain, seqnum, entity_name):
    """Build the original four-part target identifier."""
    return f"{protein}_{chain}_{seqnum}_{entity_name}"


def build_identifier_for_id_chain(protein, chain):
    """Build the grouped protein-chain identifier used in id_chain mode."""
    return f"{protein}_{chain}"


def seqnum_sort_key(value):
    """
    Sort sequence numbers numerically when possible, otherwise lexically.

    This keeps typical numeric residue identifiers in natural order while still
    handling unexpected non-numeric values safely.
    """
    cleaned_value = clean_csv_value(value)
    try:
        return (0, int(cleaned_value), cleaned_value)
    except ValueError:
        return (1, cleaned_value)


def format_entity_names(entity_names):
    """
    Return a stable display string for entity names stored in the summary CSV.

    The summary header intentionally keeps the historical ``ligands`` column
    name so the existing output shape remains stable, even when the selected
    entities are residues rather than ligands.
    """
    return ", ".join(
        sorted(
            {
                clean_csv_value(entity_name)
                for entity_name in entity_names
                if clean_csv_value(entity_name)
            }
        )
    )


def format_seqnums(seqnums):
    """Return deduplicated residue sequence numbers in a stable readable form."""
    unique_seqnums = sorted({clean_csv_value(seqnum) for seqnum in seqnums if clean_csv_value(seqnum)}, key=seqnum_sort_key)
    return ",".join(unique_seqnums)


def format_ordered_entity_names(entity_seq_pairs):
    """Return entity names ordered by their paired sequence numbers."""
    sorted_pairs = sorted(
        entity_seq_pairs,
        key=lambda pair: (seqnum_sort_key(pair["seqnum"]), pair["row_order"]),
    )
    return ",".join(pair["entity_name"] for pair in sorted_pairs)


def resolve_column_name(columns, requested_name):
    """
    Resolve a CSV column name with exact-match first, then case-insensitive match.

    This keeps the interface strict enough for production use while still being
    tolerant of minor header-case differences.
    """
    if not requested_name:
        return None
    if requested_name in columns:
        return requested_name

    case_insensitive_matches = [
        column for column in columns if column.lower() == requested_name.lower()
    ]
    if len(case_insensitive_matches) == 1:
        return case_insensitive_matches[0]
    if len(case_insensitive_matches) > 1:
        raise ValueError(
            f"Column name '{requested_name}' is ambiguous. Matching columns: "
            f"{', '.join(case_insensitive_matches)}"
        )
    return None


def resolve_optional_column(columns, role_name, explicit_name=None, candidate_names=None):
    """
    Resolve an optional semantic column.

    Explicit CLI overrides remain strict. Candidate-based discovery is lenient
    because id_chain mode can still run without summary-only metadata columns.
    """
    if explicit_name:
        return resolve_required_column(
            columns,
            role_name=role_name,
            explicit_name=explicit_name,
            candidate_names=candidate_names,
        )

    for candidate in candidate_names or []:
        resolved_candidate = resolve_column_name(columns, candidate)
        if resolved_candidate is not None:
            return resolved_candidate

    return None


def resolve_required_column(columns, role_name, explicit_name=None, candidate_names=None):
    """
    Resolve the CSV column that plays a required semantic role.

    Resolution order:
    1. explicit CLI override, if provided
    2. first available candidate from the preferred fallback list

    A clear error is raised when no valid column can be resolved.
    """
    if explicit_name:
        resolved_explicit = resolve_column_name(columns, explicit_name)
        if resolved_explicit is None:
            raise ValueError(
                f"Requested {role_name} column '{explicit_name}' was not found. "
                f"Available columns: {', '.join(columns)}"
            )
        return resolved_explicit

    for candidate in candidate_names or []:
        resolved_candidate = resolve_column_name(columns, candidate)
        if resolved_candidate is not None:
            return resolved_candidate

    candidate_text = ", ".join(candidate_names or [])
    raise ValueError(
        f"Could not resolve the required {role_name} column. "
        f"Tried: {candidate_text}. Available columns: {', '.join(columns)}"
    )


def resolve_input_columns(
    df,
    mode,
    entity_type,
    protein_column=None,
    chain_column=None,
    entity_name_column=None,
    seqnum_column=None,
):
    """
    Resolve CSV columns into the semantic roles used by the processing pipeline.

    The computational core below does not care what the original header names
    are. This helper translates a concrete CSV schema into the four logical
    roles required by the selection workflow:

    * protein
    * chain
    * entity_name  (residue or ligand name)
    * seqnum

    In ``all_4_id_chain_res_seqnum`` mode, the identifier is written in the
    conceptual order protein/chain/entity_name/seqnum.

    In ``id_chain`` mode, only ``protein`` and ``chain`` are resolved from the
    CSV. Entity-name and sequence-number columns are intentionally ignored.
    """
    columns = list(df.columns)
    resolved_protein_column = resolve_required_column(
        columns,
        role_name="protein",
        explicit_name=protein_column,
        candidate_names=["protein"],
    )
    resolved_chain_column = resolve_required_column(
        columns,
        role_name="chain",
        explicit_name=chain_column,
        candidate_names=["chain"],
    )

    entity_name_candidates = ["residue", "ligand", "name", "entity_name", "carb_name"]
    seqnum_candidates = ["seqnum", "carb_id"]

    if mode == MODE_ID_CHAIN:
        return {
            "protein": resolved_protein_column,
            "chain": resolved_chain_column,
            "entity_name": resolve_optional_column(
                columns,
                role_name="entity name",
                explicit_name=entity_name_column,
                candidate_names=entity_name_candidates,
            ),
            "seqnum": resolve_optional_column(
                columns,
                role_name="sequence number",
                explicit_name=seqnum_column,
                candidate_names=seqnum_candidates,
            ),
        }

    # In explicit all-4 mode, both ligand and residue workflows use the same
    # generic name-column role plus seqnum. ``entity_type`` affects atom
    # selection later, not CSV header naming here.
    preferred_entity_override = entity_name_column
    preferred_seq_override = seqnum_column

    resolved_entity_name_column = resolve_required_column(
        columns,
        role_name="entity name",
        explicit_name=preferred_entity_override,
        candidate_names=entity_name_candidates,
    )
    resolved_seqnum_column = resolve_required_column(
        columns,
        role_name="sequence number",
        explicit_name=preferred_seq_override,
        candidate_names=seqnum_candidates,
    )

    return {
        "protein": resolved_protein_column,
        "chain": resolved_chain_column,
        "entity_name": resolved_entity_name_column,
        "seqnum": resolved_seqnum_column,
    }


def resolve_path(path_value):
    """Resolve a potentially relative path against the current working directory."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(os.getcwd(), path_value))


def resolve_output_path(path_value, results_dir=None):
    """Resolve an output path, optionally relative to the configured results directory."""
    if os.path.isabs(path_value):
        return path_value
    if results_dir:
        return os.path.abspath(os.path.join(results_dir, path_value))
    return resolve_path(path_value)


def normalize_atom_count_map(raw_counts, section_name, config_path):
    """Return an uppercase name -> integer atom-count mapping from JSON data."""
    if not isinstance(raw_counts, dict):
        raise ValueError(
            f"Atom-count config '{config_path}' section '{section_name}' must be an object."
        )

    normalized_counts = {}
    for raw_name, raw_count in raw_counts.items():
        name = clean_csv_value(raw_name).upper()
        if not name:
            raise ValueError(
                f"Atom-count config '{config_path}' section '{section_name}' contains an empty name."
            )
        try:
            atom_count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Atom-count config '{config_path}' value for '{section_name}.{raw_name}' "
                f"must be an integer."
            ) from exc
        if atom_count < 1:
            raise ValueError(
                f"Atom-count config '{config_path}' value for '{section_name}.{raw_name}' "
                "must be at least 1."
            )
        normalized_counts[name] = atom_count

    return normalized_counts


def load_atom_count_config(config_path):
    """Load ligand and residue atom-count filters from a JSON file."""
    resolved_config_path = resolve_path(config_path)
    with open(resolved_config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Atom-count config '{resolved_config_path}' must be a JSON object.")

    if "ligand" not in config or "residue" not in config:
        raise ValueError(
            f"Atom-count config '{resolved_config_path}' must contain 'ligand' and "
            "'residue' sections."
        )

    ligand_counts = normalize_atom_count_map(
        config["ligand"],
        section_name="ligand",
        config_path=resolved_config_path,
    )
    residue_counts = normalize_atom_count_map(
        config["residue"],
        section_name="residue",
        config_path=resolved_config_path,
    )
    return ligand_counts, residue_counts


def ensure_parent_dir(path_value):
    """Create the parent directory for a file path when needed."""
    directory = os.path.dirname(path_value)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_lexical_map(csv_file):
    """
    Load the lexical atom-to-sequence mapping used by the preserved key logic.

    The historical lexical files sometimes expose the atom-name column as
    ``atom`` and sometimes as ``ATOM``. Both are supported.
    """
    df = pd.read_csv(csv_file)
    if "seq" not in df.columns:
        raise ValueError(f"Missing required 'seq' column in lexical CSV: {csv_file}")

    lexical_map = {}
    if "atom" in df.columns:
        lexical_map.update(dict(zip(df["atom"], df["seq"])))
    if "ATOM" in df.columns:
        lexical_map.update(dict(zip(df["ATOM"], df["seq"])))

    if not lexical_map:
        raise ValueError(
            f"Lexical CSV '{csv_file}' must contain either an 'atom' or 'ATOM' column."
        )

    return lexical_map


def collect_grouped_entries(
    csv_file,
    mode,
    entity_type,
    protein_column=None,
    chain_column=None,
    entity_name_column=None,
    seqnum_column=None,
):
    """
    Read the input CSV and return processing targets for the requested mode.

    Only the columns required for computation are used. Extra columns are
    ignored so the script remains robust to wider input tables.

    Important:
    * In ``all_4_id_chain_res_seqnum`` mode, each resolved row becomes one or
      more explicit processing targets.
    * In ``id_chain`` mode, protein and chain define the calculation target.
      Entity-name and seqnum columns, when present, are used only to render the
      summary metadata in sequence order.
    """
    df = pd.read_csv(csv_file, dtype=str)
    resolved_columns = resolve_input_columns(
        df=df,
        mode=mode,
        entity_type=entity_type,
        protein_column=protein_column,
        chain_column=chain_column,
        entity_name_column=entity_name_column,
        seqnum_column=seqnum_column,
    )

    protein_list = []
    protein_chain_seen = set()
    id_chain_targets_by_key = {}
    processing_targets = []

    for row_index, row in df.iterrows():
        row_number = row_index + 2  # +2 because CSV headers occupy the first line.
        protein = clean_csv_value(row.get(resolved_columns["protein"]))
        chain = clean_csv_value(row.get(resolved_columns["chain"]))
        if mode == MODE_ID_CHAIN:
            if not protein or not chain:
                print(
                    f"Warning: skipping row {row_number} because protein='{protein}' "
                    f"or chain='{chain}' is missing."
                )
                continue

            protein_chain = f"{protein}_{chain}"
            if protein_chain not in protein_chain_seen:
                protein_chain_seen.add(protein_chain)
                protein_list.append(protein_chain)
                target = {
                    "mode": MODE_ID_CHAIN,
                    "protein": protein,
                    "chain": chain,
                    "entity_type": entity_type,
                    "entity_names": [],
                    "seqnums": [],
                    "identifier": build_identifier_for_id_chain(protein, chain),
                    "summary_entities": "",
                    "summary_seqnum": "",
                    "summary_entity_seq_pairs": [],
                }
                id_chain_targets_by_key[protein_chain] = target
                processing_targets.append(target)

            entity_column = resolved_columns["entity_name"]
            seqnum_column = resolved_columns["seqnum"]
            if entity_column and seqnum_column:
                entity_name = clean_csv_value(row.get(entity_column))
                seqnums = parse_seqnums(row.get(seqnum_column))
                if entity_name and seqnums:
                    target = id_chain_targets_by_key[protein_chain]
                    for seqnum in seqnums:
                        target["summary_entity_seq_pairs"].append(
                            {
                                "seqnum": seqnum,
                                "entity_name": entity_name,
                                "row_order": len(target["summary_entity_seq_pairs"]),
                            }
                        )
            continue

        entity_name = clean_csv_value(row.get(resolved_columns["entity_name"]))
        seqnums = parse_seqnums(
            row.get(resolved_columns["seqnum"]),
            row_number=row_number,
            column_name=resolved_columns["seqnum"],
        )

        if not protein or not chain or not entity_name:
            print(
                f"Warning: skipping row {row_number} because protein='{protein}', "
                f"chain='{chain}', or {resolved_columns['entity_name']}='{entity_name}' is missing."
            )
            continue

        if not seqnums:
            continue

        protein_chain = f"{protein}_{chain}"
        if protein_chain not in protein_chain_seen:
            protein_chain_seen.add(protein_chain)
            protein_list.append(protein_chain)

        if mode == MODE_ALL_4:
            # Preserve the original row-wise behavior exactly: one calculation per
            # explicit (protein, chain, entity_name, seqnum) target.
            for seqnum in seqnums:
                processing_targets.append(
                    {
                        "mode": MODE_ALL_4,
                        "protein": protein,
                        "chain": chain,
                        "entity_type": entity_type,
                        "entity_names": [entity_name],
                        "seqnums": [seqnum],
                        "identifier": build_identifier_for_all4(protein, chain, seqnum, entity_name),
                        "summary_entities": entity_name,
                        "summary_seqnum": seqnum,
                    }
                )
            continue

    if mode == MODE_ID_CHAIN:
        for target in processing_targets:
            summary_pairs = target.pop("summary_entity_seq_pairs", [])
            if summary_pairs:
                target["summary_entities"] = format_ordered_entity_names(summary_pairs)
                target["summary_seqnum"] = format_seqnums(
                    [pair["seqnum"] for pair in summary_pairs]
                )

    return protein_list, processing_targets


def write_summary_row(target, analyzer):
    """
    Build a summary row from the just-computed analyzer state.

    The summary schema is controlled here so both modes always produce the exact
    requested header and column order.
    """
    return [
        target["identifier"],
        len(analyzer.aminoAcidCode),
        len(analyzer.totalKeys),
        len(analyzer.keyFreq),
        max(analyzer.maxDistList),
        min(analyzer.maxDistList),
        target["summary_entities"],
        target["chain"],
        target["summary_seqnum"],
    ]


def deduplicate_summary_rows(rows):
    """Remove duplicate summary rows while preserving their original order."""
    seen = set()
    unique_rows = []
    for row in rows:
        row_key = tuple(row)
        if row_key in seen:
            continue
        seen.add(row_key)
        unique_rows.append(row)
    return unique_rows


def write_summary_csv(summary_path, rows):
    """Write the summary CSV with the exact requested header."""
    ensure_parent_dir(summary_path)
    with open(summary_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(rows)


class AminoAcidAnalyzer:
    """Container for atom selection state and the preserved key-calculation logic."""

    def __init__(self, dtheta, dLen, numOfLabels):
        self.dtheta = dtheta
        self.dLen = dLen
        self.numOfLabels = numOfLabels
        # These are the different labels of the atoms with their unique code.
        self.aminoAcidLabelWithCode = {}
        # Store lexical code, coordinates, and residue sequence numbers per atom.
        self.aminoAcidCode = {}
        self.xCoordinate = {}
        self.yCoordinate = {}
        self.zCoordinate = {}
        self.aminoSeqNum = {}
        # These three hold the label code of three atoms, sorted later for key generation.
        self.initAminoLabel = [0, 0, 0]
        self.sortedAminoLabel = [0, 0, 0]
        self.sortedAminoIndex = [0, 0, 0]
        # keys with their frequency
        self.keyFreq = {}
        # total number of keys and max distance list
        self.totalKeys = []
        self.maxDistList = []
        # List of protein_chain identifiers used for download scheduling.
        self.proteinList = []
        # Preserve atom labels by index so alternate-location records do not collapse duplicate atom names.
        self.atomLabelByIndex = {}
        self.skip = False

    def setDrugLexicalMap(self, lexical_map):
        """Assign a preloaded lexical mapping to this analyzer instance."""
        self.aminoAcidLabelWithCode = dict(lexical_map)

    # Code to download the data set from rcsb.org
    def downloadDataSet(self, download_dir=None):
        if download_dir is None:
            download_dir = PDB_DIR_PATH

        downloaded_files = set()
        for file in self.proteinList:
            pdb_name = str(file.split("_")[0])
            if pdb_name in downloaded_files:
                continue

            pdb_url = "https://files.rcsb.org/download/" + pdb_name + ".pdb"
            save_path = pdb_name + ".pdb"
            print(save_path)
            try:
                response = requests.get(pdb_url)
                if response.status_code == 200:
                    with open(os.path.join(download_dir, save_path), "wb") as pdb_file:
                        pdb_file.write(response.content)
                    downloaded_files.add(pdb_name)
                    print("PDB file for", file, " downloaded and saved as ", save_path, ".")
                else:
                    print("Failed to download PDB file for", file, ". Status code:", response.status_code)

            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred while downloading {file}: {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                print(f"Connection error occurred while downloading {file}: {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                print(f"Timeout error occurred while downloading {file}: {timeout_err}")
            except requests.exceptions.RequestException as req_err:
                print(f"An error occurred while downloading {file}: {req_err}")
            except Exception as e:
                print(f"An unexpected error occurred while downloading {file}: {e}")

    def thetaClass(self, theta):
        return thetaClass(theta)

    def dist12Class(self, dist12):
        return dist12Class(dist12)

    # Calculating the distance between two selected atoms.
    def calDistance(self, l1_index, l2_index):
        x1 = self.xCoordinate[l1_index]
        x2 = self.xCoordinate[l2_index]
        y1 = self.yCoordinate[l1_index]
        y2 = self.yCoordinate[l2_index]
        z1 = self.zCoordinate[l1_index]
        z2 = self.zCoordinate[l2_index]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def findTheIndex(self, l2_index, p1, q1, r1):
        if l2_index == p1:
            l1_index0 = q1
            l2_index1 = r1
        elif l2_index == q1:
            l1_index0 = p1
            l2_index1 = r1
        elif l2_index == r1:
            l1_index0 = p1
            l2_index1 = q1
        return l1_index0, l2_index1

    def _reset_state_for_calculation(self):
        """Reset per-target mutable state before loading a new atom selection."""
        self.totalKeys = []
        self.maxDistList = []
        self.keyFreq = {}
        self.atomLabelByIndex = {}
        self.aminoAcidCode = {}
        self.aminoSeqNum = {}
        self.xCoordinate = {}
        self.yCoordinate = {}
        self.zCoordinate = {}

    def _is_matching_pdb_record(self, line, chain, entity_name_set, seq_value_set, entity_type):
        """
        Decide whether a PDB line belongs to the requested selection.

        This helper is intentionally the only place where ligand-vs-residue
        selection rules differ:

        * ligand  -> ``HETATM`` records only, and hydrogen/deuterium atoms are excluded
        * residue -> ``ATOM`` records only, and hydrogen/deuterium atoms are excluded
        """
        record_type = line[0:6].rstrip()
        if record_type not in {"ATOM", "HETATM"}:
            return False

        if line[16:17].strip() not in ("", "A"):
            return False

        if line[21:22].strip() != chain:
            return False

        residue_name = line[17:20].strip()
        residue_seq = line[22:27].strip()

        # In id_chain mode the caller intentionally passes empty filters so the
        # selection covers the whole chain. In explicit all-4 mode, both the
        # entity name and seqnum must match the requested target.
        if entity_name_set and residue_name not in entity_name_set:
            return False
        if seq_value_set and residue_seq not in seq_value_set:
            return False

        if entity_type == ENTITY_LIGAND:
            return (
                record_type == "HETATM"
                and line[77:80].strip() != "H"
                and line[77:80].strip() != "D"
            )

        return (
            record_type == "ATOM"
            and line[77:80].strip() != "H"
            and line[77:80].strip() != "D"
        )

    def _store_selected_atom(self, incrementVal, atom_label, line):
        """Store one already-filtered atom in the analyzer state."""
        self.aminoAcidCode[incrementVal] = int(self.aminoAcidLabelWithCode[atom_label])
        self.aminoSeqNum[incrementVal] = str(line[22:27])
        self.xCoordinate[incrementVal] = float(line[30:38])
        self.yCoordinate[incrementVal] = float(line[38:46])
        self.zCoordinate[incrementVal] = float(line[46:54])
        self.atomLabelByIndex[incrementVal] = atom_label
        return incrementVal + 1

    def _passes_ligand_atom_count_filter(self, ligand_name, atom_count, output_identifier):
        """Permit ligand instances only when their atom count matches the configured count."""
        expected_atom_count = EXPECTED_LIGAND_ATOM_COUNTS.get(ligand_name)
        if expected_atom_count is not None and atom_count == expected_atom_count:
            return True

        expected_text = (
            str(expected_atom_count)
            if expected_atom_count is not None
            else "a configured ligand atom count"
        )
        print(
            f"Warning: skipping {output_identifier} because ligand {ligand_name} "
            f"has {atom_count} matching atom(s), expected {expected_text}."
        )
        return False

    def _passes_residue_instance_atom_count_filter(self, residue_name, atom_count, output_identifier):
        """Permit residue instances only when their atom count matches the configured count."""
        expected_atom_count = EXPECTED_RESIDUE_ATOM_COUNTS.get(residue_name)
        if expected_atom_count is not None and atom_count == expected_atom_count:
            return True

        expected_text = (
            str(expected_atom_count)
            if expected_atom_count is not None
            else "a configured residue atom count"
        )
        print(
            f"Warning: skipping {output_identifier} because residue {residue_name} "
            f"has {atom_count} matching atom(s), expected {expected_text}."
        )
        return False

    def _load_atoms_for_selection(self, fileName, chain, entity_names, seq_values, entity_type, output_identifier=None):
        """
        Load atoms matching the requested chain, entity names, and sequence numbers.

        Atom-selection behavior depends on ``entity_type`` only. All distance,
        theta, and key-generation logic remains unchanged after the atom set is
        loaded into memory.
        """
        entity_name_set = {clean_csv_value(name) for name in entity_names if clean_csv_value(name)}
        seq_value_set = {clean_csv_value(value) for value in seq_values if clean_csv_value(value)}
        incrementVal = 0
        missing_lexical_atoms = set()
        entity_atom_groups = {}
        entity_group_order = []

        with open(os.path.join(PDB_DIR_PATH, fileName + ".pdb"), "r") as pdbFile:
            for line in pdbFile:
                try:
                    # Keep scanning after TER because carbohydrate HETATM records
                    # can appear later in the same chain.
                    if line[0:6].rstrip() == "ENDMDL":
                        break
                    if line[0:6].rstrip() == "MODEL" and int(line[10:14].rstrip()) > 1:
                        break

                    if self._is_matching_pdb_record(
                        line=line,
                        chain=chain,
                        entity_name_set=entity_name_set,
                        seq_value_set=seq_value_set,
                        entity_type=entity_type,
                    ):
                        atom_label = line[13:16].rstrip()
                        if atom_label not in self.aminoAcidLabelWithCode:
                            missing_lexical_atoms.add(atom_label)
                            continue

                        entity_name = line[17:20].strip().upper()
                        entity_seq = line[22:27].strip()
                        entity_key = (entity_name, entity_seq)
                        if entity_key not in entity_atom_groups:
                            entity_atom_groups[entity_key] = []
                            entity_group_order.append(entity_key)
                        entity_atom_groups[entity_key].append((atom_label, line))
                except Exception as e:
                    print("Their is an error in: ", line, pdbFile)
                    print(e)

        for entity_name, entity_seq in entity_group_order:
            entity_atoms = entity_atom_groups[(entity_name, entity_seq)]
            entity_identifier = f"{fileName}_{chain}_{entity_seq}_{entity_name}"
            if entity_type == ENTITY_LIGAND:
                passes_filter = self._passes_ligand_atom_count_filter(
                    ligand_name=entity_name,
                    atom_count=len(entity_atoms),
                    output_identifier=entity_identifier,
                )
            else:
                passes_filter = self._passes_residue_instance_atom_count_filter(
                    residue_name=entity_name,
                    atom_count=len(entity_atoms),
                    output_identifier=entity_identifier,
                )

            if not passes_filter:
                continue

            for atom_label, line in entity_atoms:
                incrementVal = self._store_selected_atom(incrementVal, atom_label, line)

        if missing_lexical_atoms:
            print(
                f"Warning: skipped {len(missing_lexical_atoms)} atom label(s) for "
                f"{fileName}_{chain} because they are missing from the lexical map: "
                f"{', '.join(sorted(missing_lexical_atoms))}"
            )

    def _passes_residue_atom_count_filter(self, entity_type, entity_names, seq_values, output_identifier):
        """
        Compatibility hook retained for the calculation wrapper.

        Residue atom-count filtering now happens per residue instance during atom
        loading so grouped id_chain targets can drop only incomplete residues
        while preserving complete residues in the same protein-chain selection.
        """
        return True

    def _write_key_outputs(self, output_identifier, protein_path=None):
        """
        Write triplet and frequency files using the preserved key-calculation logic.
        """
        if protein_path is None:
            protein_path = PROTEIN_DIR_PATH

        triplets_path = os.path.join(protein_path, f"{output_identifier}.keys_theta29_dist18")
        key_freq_path = os.path.join(protein_path, f"{output_identifier}.keys_Freq_theta29_dist18")

        with open(triplets_path, "w") as tripletsFile, open(key_freq_path, "w") as keyFreqFile:
            # This is the four rules that calculates the label, theta, and key
            # (3 atoms form a triplet). The logic in this block is preserved.
            for i in range(0, len(self.aminoAcidCode) - 2):
                for j in range(i + 1, len(self.aminoAcidCode) - 1):
                    for k in range(j + 1, len(self.aminoAcidCode)):
                        # This is a dictionary to keep the index and the labels
                        labelIndexToUse = {}
                        # First, Second and Third label and Index
                        labelIndexToUse[self.aminoAcidCode[i]] = i
                        labelIndexToUse[self.aminoAcidCode[j]] = j
                        labelIndexToUse[self.aminoAcidCode[k]] = k
                        # First, Second and Third amino label list
                        self.initAminoLabel[0] = self.aminoAcidCode[i]
                        self.initAminoLabel[1] = self.aminoAcidCode[j]
                        self.initAminoLabel[2] = self.aminoAcidCode[k]
                        # Sorted labels from above list
                        sortedAminoLabel = list(self.initAminoLabel)
                        # Reverse order from above sorted list
                        sortedAminoLabel.sort(reverse=True)

                        # The fourth case when l1=l2=l3
                        if (sortedAminoLabel[0] == sortedAminoLabel[1]) and (sortedAminoLabel[1] == sortedAminoLabel[2]):
                            distance1_2 = self.calDistance(i, j)
                            distance1_3 = self.calDistance(i, k)
                            distance2_3 = self.calDistance(j, k)
                            if distance1_2 >= (max(distance1_2, distance1_3, distance2_3)):
                                l1_index0 = i
                                l2_index1 = j
                                l3_index2 = k
                            elif distance1_3 >= (max(distance1_2, distance1_3, distance2_3)):
                                l1_index0 = i
                                l2_index1 = k
                                l3_index2 = j
                            else:
                                l1_index0 = j
                                l2_index1 = k
                                l3_index2 = i

                        # Third condition when l1=l2>l3
                        elif (sortedAminoLabel[0] == sortedAminoLabel[1]) and (sortedAminoLabel[1] != sortedAminoLabel[2]):
                            l3_index2 = labelIndexToUse[sortedAminoLabel[2]]
                            indices = self.findTheIndex(l3_index2, i, j, k)
                            first = l3_index2
                            second = indices[0]
                            third = indices[1]
                            distance1_3 = self.calDistance(second, first)
                            distance2_3 = self.calDistance(third, first)
                            if distance1_3 >= distance2_3:
                                l1_index0 = indices[0]
                                l2_index1 = indices[1]
                            else:
                                l1_index0 = indices[1]
                                l2_index1 = indices[0]

                        # Second condition when l1>l2=l3
                        elif (sortedAminoLabel[0] != sortedAminoLabel[1]) and (sortedAminoLabel[1] == sortedAminoLabel[2]):
                            l1_index0 = labelIndexToUse[sortedAminoLabel[0]]
                            indices = self.findTheIndex(l1_index0, i, j, k)
                            if self.calDistance(l1_index0, indices[0]) >= self.calDistance(l1_index0, indices[1]):
                                l2_index1 = indices[0]
                                l3_index2 = indices[1]
                            else:
                                l3_index2 = indices[0]
                                l2_index1 = indices[1]

                        # First condition when l1!=l2!=l3
                        elif (
                            (sortedAminoLabel[0] != sortedAminoLabel[1])
                            and (sortedAminoLabel[0] != sortedAminoLabel[2])
                            and (sortedAminoLabel[1] != sortedAminoLabel[2])
                        ):
                            # Getting the index from the labelIndexToUse from sortedAminoLabel use
                            for index in range(0, 3):
                                self.sortedAminoIndex[index] = labelIndexToUse[sortedAminoLabel[index]]
                            l1_index0 = self.sortedAminoIndex[0]
                            l2_index1 = self.sortedAminoIndex[1]
                            l3_index2 = self.sortedAminoIndex[2]

                        distance01 = self.calDistance(l1_index0, l2_index1)
                        # Calculating the mid distance
                        midDis01 = distance01 / 2
                        distance02 = self.calDistance(l1_index0, l3_index2)
                        distance12 = self.calDistance(l2_index1, l3_index2)
                        # Calculating the max distance (D)
                        maxDistance = max(distance01, distance02, distance12)
                        # Calculating the mid point
                        m1 = (self.xCoordinate[l1_index0] + self.xCoordinate[l2_index1]) / 2
                        m2 = (self.yCoordinate[l1_index0] + self.yCoordinate[l2_index1]) / 2
                        m3 = (self.zCoordinate[l1_index0] + self.zCoordinate[l2_index1]) / 2

                        # Calculating the d3 distance
                        d3 = math.sqrt((m1 - self.xCoordinate[l3_index2]) ** 2 + (m2 - self.yCoordinate[l3_index2]) ** 2 + (m3 - self.zCoordinate[l3_index2]) ** 2)

                        # Calculating thetaAngle1
                        thetaAngle1 = 180 * (math.acos((distance02 ** 2 - midDis01 ** 2 - d3 ** 2) / (2 * midDis01 * d3))) / 3.14

                        # Check in which category does the angle falls
                        if thetaAngle1 <= 90:
                            theta = thetaAngle1
                        else:
                            theta = abs(180 - thetaAngle1)

                        # Calculating the bin values for theta and max distance
                        binTheta = self.thetaClass(theta)
                        binLength = self.dist12Class(maxDistance)

                        aminoAcidR1 = self.atomLabelByIndex[l1_index0]
                        aminoAcidR2 = self.atomLabelByIndex[l2_index1]
                        aminoAcidR3 = self.atomLabelByIndex[l3_index2]
                        # These are the sequence number of the three amino acids
                        seqNumber1 = list(self.aminoSeqNum.values())[l1_index0]
                        seqNumber2 = list(self.aminoSeqNum.values())[l2_index1]
                        seqNumber3 = list(self.aminoSeqNum.values())[l3_index2]

                        # These are the coordinates of the three amino acids
                        aminoAcidC10, aminoAcidC11, aminoAcidC12 = self.xCoordinate[l1_index0], self.yCoordinate[l1_index0], self.zCoordinate[l1_index0]
                        aminoAcidC20, aminoAcidC21, aminoAcidC22 = self.xCoordinate[l2_index1], self.yCoordinate[l2_index1], self.zCoordinate[l2_index1]
                        aminoAcidC30, aminoAcidC31, aminoAcidC32 = self.xCoordinate[l3_index2], self.yCoordinate[l3_index2], self.zCoordinate[l3_index2]

                        # Calculating the triplets key value
                        tripletKeys = (
                            dLen * dtheta * (numOfLabels ** 2) * (self.aminoAcidCode[l1_index0] - 1)
                            + dLen * dtheta * (numOfLabels) * (self.aminoAcidCode[l2_index1] - 1)
                            + dLen * dtheta * (self.aminoAcidCode[l3_index2] - 1)
                            + dtheta * (binLength - 1)
                            + (binTheta - 1)
                        )

                        # Total number of keys and max distance list
                        self.totalKeys.append(tripletKeys)
                        self.maxDistList.append(maxDistance)

                        # Filtering out the distinct keys
                        if tripletKeys in self.keyFreq:
                            self.keyFreq[tripletKeys] += 1
                        else:
                            self.keyFreq[tripletKeys] = 1

                        # These are the info of all the triplets
                        tripletInfoAll = (
                            str(tripletKeys)
                            + "\t"
                            + str(aminoAcidR1)
                            + "\t"
                            + str(seqNumber1)
                            + "\t"
                            + str(aminoAcidR2)
                            + "\t"
                            + str(seqNumber2)
                            + "\t"
                            + str(aminoAcidR3)
                            + "\t"
                            + str(seqNumber3)
                            + "\t"
                            + str(binTheta)
                            + "\t"
                            + str(theta)
                            + "\t"
                            + str(binLength)
                            + "\t"
                            + str(maxDistance)
                            + "\t"
                            + str(aminoAcidC10)
                            + "\t"
                            + str(aminoAcidC11)
                            + "\t"
                            + str(aminoAcidC12)
                            + "\t"
                            + str(aminoAcidC20)
                            + "\t"
                            + str(aminoAcidC21)
                            + "\t"
                            + str(aminoAcidC22)
                            + "\t"
                            + str(aminoAcidC30)
                            + "\t"
                            + str(aminoAcidC31)
                            + "\t"
                            + str(aminoAcidC32)
                            + "\n"
                        )
                        tripletsFile.writelines(tripletInfoAll)

            # Storing the distinct keys in a file
            for values in self.keyFreq:
                keyFreqFile.writelines([str(values), "\t", str(self.keyFreq[values]), "\n"])

    def calculate_for_selection(self, fileName, chain, seq_values, entity_names, entity_type, output_identifier, protein_path=None):
        """
        Shared wrapper that prepares the selected atom set before running the preserved key logic.

        The only configurable behavior here is the atom-selection layer:
        ``entity_names`` + ``seq_values`` define the target scope, and
        ``entity_type`` chooses ligand-heavy-atom vs residue-atom record rules.
        The key-generation mathematics that follow are intentionally unchanged.
        """
        if protein_path is None:
            protein_path = PROTEIN_DIR_PATH

        self._reset_state_for_calculation()
        self._load_atoms_for_selection(
            fileName=fileName,
            chain=chain,
            entity_names=entity_names,
            seq_values=seq_values,
            entity_type=entity_type,
            output_identifier=output_identifier,
        )

        if not self._passes_residue_atom_count_filter(
            entity_type=entity_type,
            entity_names=entity_names,
            seq_values=seq_values,
            output_identifier=output_identifier,
        ):
            return False

        if len(self.aminoAcidCode) < 3:
            print(
                f"Warning: skipping {output_identifier} because only "
                f"{len(self.aminoAcidCode)} matching atoms were found."
            )
            return False

        self._write_key_outputs(output_identifier, protein_path=protein_path)
        return True

    # This function is preserved as the original single-residue entry point.
    def calcuTheteAndKey(self, fileName, chain, seq_value, entity_name, entity_type, protein_path=None):
        output_identifier = build_identifier_for_all4(fileName, chain, seq_value, entity_name)
        return self.calculate_for_selection(
            fileName=fileName,
            chain=chain,
            seq_values=[seq_value],
            entity_names=[entity_name],
            entity_type=entity_type,
            output_identifier=output_identifier,
            protein_path=protein_path,
        )

    def calcuTheteAndKeyForGroupedSelection(self, fileName, chain, seq_values, entity_names, entity_type, output_identifier, protein_path=None):
        """
        Grouped-mode wrapper.

        In id_chain mode the CSV contributes only protein and chain. Passing
        empty ``entity_names`` and ``seq_values`` tells the selector to use the
        full matching chain scope for the chosen entity type.
        """
        return self.calculate_for_selection(
            fileName=fileName,
            chain=chain,
            seq_values=seq_values,
            entity_names=entity_names,
            entity_type=entity_type,
            output_identifier=output_identifier,
            protein_path=protein_path,
        )


def process_target(target, lexical_map, ligand_atom_counts, residue_atom_counts):
    """
    Process a single target in a worker-local analyzer instance.

    Using a fresh analyzer per task keeps mutable state isolated while preserving
    the exact same atom-selection and key-generation code path.
    """
    global EXPECTED_LIGAND_ATOM_COUNTS, EXPECTED_RESIDUE_ATOM_COUNTS
    EXPECTED_LIGAND_ATOM_COUNTS = ligand_atom_counts
    EXPECTED_RESIDUE_ATOM_COUNTS = residue_atom_counts

    analyzer = AminoAcidAnalyzer(dtheta, dLen, numOfLabels)
    analyzer.setDrugLexicalMap(lexical_map)

    if target["mode"] == MODE_ALL_4:
        processed = analyzer.calcuTheteAndKey(
            target["protein"],
            target["chain"],
            target["seqnums"][0],
            target["entity_names"][0],
            target["entity_type"],
        )
    else:
        processed = analyzer.calcuTheteAndKeyForGroupedSelection(
            fileName=target["protein"],
            chain=target["chain"],
            seq_values=target["seqnums"],
            entity_names=target["entity_names"],
            entity_type=target["entity_type"],
            output_identifier=target["identifier"],
        )

    if not processed:
        return None

    return write_summary_row(target, analyzer)


def main(
    input_path,
    csv_file,
    lexical_file,
    output_path,
    summary_csv,
    results_dir,
    dtheta_value,
    dlen_value,
    num_labels_value,
    num_cores,
    mode,
    entity_type,
    protein_column=None,
    chain_column=None,
    entity_name_column=None,
    seqnum_column=None,
    atom_count_config=None,
):
    """Main entry point for both CLI usage and batch execution."""
    global CSV_FILE_PATH, CSV_FILE_LEXICAL_PATH, PDB_DIR_PATH, PROTEIN_DIR_PATH, SUMMARY_CSV_PATH, dtheta, dLen, numOfLabels, EXPECTED_LIGAND_ATOM_COUNTS, EXPECTED_RESIDUE_ATOM_COUNTS

    if num_cores < 1:
        raise ValueError("--num_cores must be at least 1.")

    normalized_mode = normalize_mode(mode)
    normalized_entity_type = normalize_entity_type(entity_type)
    resolved_results_dir = resolve_path(results_dir) if results_dir else None
    CSV_FILE_PATH = resolve_path(csv_file)
    CSV_FILE_LEXICAL_PATH = resolve_path(lexical_file)
    PDB_DIR_PATH = resolve_output_path(input_path, resolved_results_dir)
    PROTEIN_DIR_PATH = resolve_output_path(output_path, resolved_results_dir)
    SUMMARY_CSV_PATH = resolve_output_path(summary_csv, resolved_results_dir)
    dtheta = dtheta_value
    dLen = dlen_value
    numOfLabels = num_labels_value
    atom_count_config_path = atom_count_config or DEFAULT_ATOM_COUNT_CONFIG
    EXPECTED_LIGAND_ATOM_COUNTS, EXPECTED_RESIDUE_ATOM_COUNTS = load_atom_count_config(
        atom_count_config_path
    )

    if resolved_results_dir:
        os.makedirs(resolved_results_dir, exist_ok=True)
    os.makedirs(PDB_DIR_PATH, exist_ok=True)
    os.makedirs(PROTEIN_DIR_PATH, exist_ok=True)
    ensure_parent_dir(SUMMARY_CSV_PATH)

    protein_list, processing_targets = collect_grouped_entries(
        csv_file=CSV_FILE_PATH,
        mode=normalized_mode,
        entity_type=normalized_entity_type,
        protein_column=protein_column,
        chain_column=chain_column,
        entity_name_column=entity_name_column,
        seqnum_column=seqnum_column,
    )
    lexical_map = load_lexical_map(CSV_FILE_LEXICAL_PATH)

    if not processing_targets:
        print("Warning: no valid processing targets were found in the input CSV.")
        write_summary_csv(SUMMARY_CSV_PATH, [])
        return

    # Keep the original download step, but schedule it from the mode-aware target list.
    download_analyzer = AminoAcidAnalyzer(dtheta, dLen, numOfLabels)
    download_analyzer.proteinList = protein_list
    download_analyzer.downloadDataSet()

    results = Parallel(n_jobs=num_cores, verbose=50)(
        delayed(process_target)(
            target,
            lexical_map,
            EXPECTED_LIGAND_ATOM_COUNTS,
            EXPECTED_RESIDUE_ATOM_COUNTS,
        )
        for target in processing_targets
    )

    summary_rows = deduplicate_summary_rows([row for row in results if row is not None])
    write_summary_csv(SUMMARY_CSV_PATH, summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate atom-triplet keys using either explicit all-4 targets or "
            "grouped protein-chain targets. CSV columns are resolved using "
            "protein, chain, one generic name column, and seqnum."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--input_path", type=str, required=True, help="Directory where PDB files will be stored")
    parser.add_argument("-c", "--csv_file", type=str, required=True, help="Path to the target CSV file")
    parser.add_argument("-l", "--lexical_file", type=str, required=True, help="Path to the lexical CSV file")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Directory where key files will be written")
    parser.add_argument(
        "-s",
        "--summary_csv",
        type=str,
        default="proteinNumKeysDist.csv",
        help="Path to the summary CSV file (stored under results_dir when a relative path is provided)",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default=None,
        help="Optional base directory under which input_path, output_path, and summary_csv will be created when relative paths are used",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=MODE_ALL_4,
        choices=[MODE_ALL_4, MODE_ID_CHAIN],
        help=(
            "Calculation mode. "
            f"Use '{MODE_ALL_4}' to process one explicit target per resolved "
            "protein/chain/entity_name/seqnum combination, "
            f"or '{MODE_ID_CHAIN}' to use only protein+chain from the CSV and "
            "run the grouped calculation across the full matching chain scope."
        ),
    )
    parser.add_argument(
        "--entity_type",
        type=str,
        default=ENTITY_LIGAND,
        choices=[ENTITY_LIGAND, ENTITY_RESIDUE],
        help=(
            "Controls only the atom-selection stage. "
            "'ligand' preserves the previous heavy-atom HETATM behavior. "
            "'residue' selects matching ATOM/HETATM records."
        ),
    )
    parser.add_argument(
        "--protein_column",
        type=str,
        default=None,
        help="Optional CSV column name to use as the protein identifier role.",
    )
    parser.add_argument(
        "--chain_column",
        type=str,
        default=None,
        help="Optional CSV column name to use as the chain identifier role.",
    )
    parser.add_argument(
        "--entity_name_column",
        type=str,
        default=None,
        help=(
            "Optional override for the generic entity-name role. "
            "Use this when the target name column is not one of the default "
            "headers such as 'residue', 'ligand', 'name', or 'entity_name'."
        ),
    )
    parser.add_argument(
        "--seqnum_column",
        type=str,
        default=None,
        help=(
            "Optional sequence-number column override. Use this when the CSV does "
            "not expose the role as 'seqnum'."
        ),
    )
    parser.add_argument(
        "--atom_count_config",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing ligand and residue expected atom counts. "
            "Defaults to expected_atom_counts.json next to this script."
        ),
    )
    parser.add_argument("--dtheta", type=int, default=29, help="Theta bin count parameter used by the preserved key logic")
    parser.add_argument("--dlen", type=int, default=18, help="Distance bin count parameter used by the preserved key logic")
    parser.add_argument("--num_labels", type=int, default=112, help="Number of lexical labels used in key generation")
    parser.add_argument(
        "--num_cores",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPU cores used by joblib Parallel",
    )
    args = parser.parse_args()
    main(
        args.input_path,
        args.csv_file,
        args.lexical_file,
        args.output_path,
        args.summary_csv,
        args.results_dir,
        args.dtheta,
        args.dlen,
        args.num_labels,
        args.num_cores,
        args.mode,
        args.entity_type,
        args.protein_column,
        args.chain_column,
        args.entity_name_column,
        args.seqnum_column,
        args.atom_count_config,
    )
