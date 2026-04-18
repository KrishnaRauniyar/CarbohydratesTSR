"""
Generate carbohydrate atom-triplet keys from a CSV of carbohydrate residue targets.

This module supports two calculation modes that share the exact same triplet/key
generation core:

1. ``all_4_id_chain_res_seqnum``
   Preserves the existing behavior. Each CSV row is interpreted using the
   explicit ``protein``, ``chain``, ``carb_name``, and ``carb_id`` values.
   When ``carb_id`` contains multiple sequence numbers such as ``1;2``, the
   script expands that row into one calculation per sequence number and writes
   output files named like ``7E6X_E_1_NAG.keys_theta29_dist18``.

2. ``id_chain``
   Groups CSV rows by ``protein`` and ``chain`` only. For each protein-chain
   pair, the script merges all unique ligand names and all unique residue
   sequence numbers from the matching rows, selects the combined heavy-atom
   set from the PDB file, and then runs the unchanged key logic on that merged
   atom set. Output files are written with names like
   ``10OP_Y.keys_theta29_dist18``.
"""

import argparse
import csv
import math
import multiprocessing
import os
import sys

import pandas as pd
import requests
from joblib import Parallel, delayed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_GENERATOR_DIR = os.path.dirname(SCRIPT_DIR)
for import_path in (SCRIPT_DIR, KEY_GENERATOR_DIR):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

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


def parse_seqnums(value, row_number=None):
    """
    Parse the ``carb_id`` column into a list of individual residue sequence numbers.

    The source file may contain a single residue sequence number such as ``3`` or
    a semicolon-delimited list such as ``4;5``. Empty fragments are ignored.
    """
    seqnums = []
    raw_value = clean_csv_value(value)
    if not raw_value:
        if row_number is not None:
            print(f"Warning: skipping row {row_number} because carb_id is missing.")
        return seqnums

    for seq_value in raw_value.split(";"):
        cleaned_value = seq_value.strip()
        if cleaned_value:
            seqnums.append(cleaned_value)

    if not seqnums and row_number is not None:
        print(
            f"Warning: skipping row {row_number} because carb_id '{raw_value}' "
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


def build_identifier_for_all4(protein, chain, seqnum, ligand):
    """Build the original residue-specific identifier."""
    return f"{protein}_{chain}_{seqnum}_{ligand}"


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


def format_ligands(ligands):
    """Return a stable, human-readable ligand list for the summary CSV."""
    return ", ".join(sorted({clean_csv_value(ligand) for ligand in ligands if clean_csv_value(ligand)}))


def format_seqnums(seqnums):
    """Return deduplicated residue sequence numbers in a stable readable form."""
    unique_seqnums = sorted({clean_csv_value(seqnum) for seqnum in seqnums if clean_csv_value(seqnum)}, key=seqnum_sort_key)
    return ",".join(unique_seqnums)


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


def collect_grouped_entries(csv_file, mode):
    """
    Read the sample-detail CSV and return processing targets for the requested mode.

    Only the columns required for computation are used. Extra columns such as
    ``group`` are ignored so the script remains robust to wider input tables.
    """
    df = pd.read_csv(csv_file, dtype=str)
    required_columns = {"protein", "chain", "carb_name", "carb_id"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in input CSV: {missing_list}")

    protein_list = []
    protein_chain_seen = set()
    processing_targets = []

    # ``grouped_targets`` is only used in id_chain mode. A regular dict is enough
    # because Python preserves insertion order, which keeps group order stable.
    grouped_targets = {}

    for row_index, row in df.iterrows():
        row_number = row_index + 2  # +2 because CSV headers occupy the first line.
        protein = clean_csv_value(row.get("protein"))
        chain = clean_csv_value(row.get("chain"))
        carb_name = clean_csv_value(row.get("carb_name"))
        seqnums = parse_seqnums(row.get("carb_id"), row_number=row_number)

        if not protein or not chain or not carb_name:
            print(
                f"Warning: skipping row {row_number} because protein='{protein}', "
                f"chain='{chain}', or carb_name='{carb_name}' is missing."
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
            # explicit (protein, chain, carb_name, seqnum) target.
            for seqnum in seqnums:
                processing_targets.append(
                    {
                        "mode": MODE_ALL_4,
                        "protein": protein,
                        "chain": chain,
                        "ligands": [carb_name],
                        "seqnums": [seqnum],
                        "identifier": build_identifier_for_all4(protein, chain, seqnum, carb_name),
                        "summary_ligands": carb_name,
                        "summary_seqnum": seqnum,
                    }
                )
            continue

        group_key = (protein, chain)
        if group_key not in grouped_targets:
            grouped_targets[group_key] = {
                "protein": protein,
                "chain": chain,
                "ligands": set(),
                "seqnums": set(),
            }

        grouped_targets[group_key]["ligands"].add(carb_name)
        grouped_targets[group_key]["seqnums"].update(seqnums)

    if mode == MODE_ID_CHAIN:
        for grouped_target in grouped_targets.values():
            sorted_ligands = sorted(grouped_target["ligands"])
            sorted_seqnums = sorted(grouped_target["seqnums"], key=seqnum_sort_key)
            protein = grouped_target["protein"]
            chain = grouped_target["chain"]
            processing_targets.append(
                {
                    "mode": MODE_ID_CHAIN,
                    "protein": protein,
                    "chain": chain,
                    "ligands": sorted_ligands,
                    "seqnums": sorted_seqnums,
                    "identifier": build_identifier_for_id_chain(protein, chain),
                    "summary_ligands": format_ligands(sorted_ligands),
                    "summary_seqnum": format_seqnums(sorted_seqnums),
                }
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
        target["summary_ligands"],
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

    def _load_atoms_for_selection(self, fileName, chain, residue_names, seq_values):
        """
        Load heavy atoms matching the requested chain plus allowed residue filters.

        This is the only place where the new mode changes behavior: the preserved
        triplet/key logic still runs on whatever atom set this method selects.
        """
        residue_name_set = set(residue_names)
        seq_value_set = set(seq_values)
        incrementVal = 0

        with open(os.path.join(PDB_DIR_PATH, fileName + ".pdb"), "r") as pdbFile:
            for line in pdbFile:
                try:
                    # Keep scanning after TER because carbohydrate HETATM records
                    # can appear later in the same chain.
                    if line[0:6].rstrip() == "ENDMDL":
                        break
                    if line[0:6].rstrip() == "MODEL" and int(line[10:14].rstrip()) > 1:
                        break

                    residue_name = line[17:20].strip()
                    residue_seq = line[22:27].strip()

                    if (
                        line.startswith("HETATM")
                        and line[16:17].strip() in ("", "A")  # accept blank or "A" alternate locations
                        and line[21:22].strip() == chain
                        and residue_name in residue_name_set
                        and residue_seq in seq_value_set
                        and line[77:80].strip() != "H"
                        and line[77:80].strip() != "D"
                    ):
                        # Reading the lines in pdb file and then assigning residue atom to its lexical value.
                        self.aminoAcidCode[incrementVal] = int(self.aminoAcidLabelWithCode[line[13:16].rstrip()])
                        # This is the residue sequence number stored for output/reporting.
                        self.aminoSeqNum[incrementVal] = str(line[22:27])
                        self.xCoordinate[incrementVal] = float(line[30:38])
                        self.yCoordinate[incrementVal] = float(line[38:46])
                        self.zCoordinate[incrementVal] = float(line[46:54])
                        self.atomLabelByIndex[incrementVal] = line[13:16].rstrip()
                        incrementVal += 1
                except Exception as e:
                    print("Their is an error in: ", line, pdbFile)
                    print(e)

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

    def calculate_for_selection(self, fileName, chain, seq_values, residue_names, output_identifier, protein_path=None):
        """
        Shared wrapper that prepares the selected atom set before running the preserved key logic.

        The only behavioral difference between modes is which ``seq_values`` and
        ``residue_names`` get passed into this wrapper.
        """
        if protein_path is None:
            protein_path = PROTEIN_DIR_PATH

        self._reset_state_for_calculation()
        self._load_atoms_for_selection(fileName, chain, residue_names, seq_values)

        if len(self.aminoAcidCode) < 3:
            print(
                f"Warning: skipping {output_identifier} because only "
                f"{len(self.aminoAcidCode)} matching non-hydrogen atoms were found."
            )
            return False

        self._write_key_outputs(output_identifier, protein_path=protein_path)
        return True

    # This function is preserved as the original single-residue entry point.
    def calcuTheteAndKey(self, fileName, chain, seq_value, chain_identity, protein_path=None):
        output_identifier = build_identifier_for_all4(fileName, chain, seq_value, chain_identity)
        return self.calculate_for_selection(
            fileName=fileName,
            chain=chain,
            seq_values=[seq_value],
            residue_names=[chain_identity],
            output_identifier=output_identifier,
            protein_path=protein_path,
        )

    def calcuTheteAndKeyForGroupedSelection(self, fileName, chain, seq_values, chain_identities, output_identifier, protein_path=None):
        """
        Grouped-mode wrapper.

        It still reuses the exact same key logic; it only changes the selected
        atom scope by passing merged residue names and sequence numbers.
        """
        return self.calculate_for_selection(
            fileName=fileName,
            chain=chain,
            seq_values=seq_values,
            residue_names=chain_identities,
            output_identifier=output_identifier,
            protein_path=protein_path,
        )


def process_target(target, lexical_map):
    """
    Process a single target in a worker-local analyzer instance.

    Using a fresh analyzer per task keeps mutable state isolated while preserving
    the exact same atom-selection and key-generation code path.
    """
    analyzer = AminoAcidAnalyzer(dtheta, dLen, numOfLabels)
    analyzer.setDrugLexicalMap(lexical_map)

    if target["mode"] == MODE_ALL_4:
        processed = analyzer.calcuTheteAndKey(
            target["protein"],
            target["chain"],
            target["seqnums"][0],
            target["ligands"][0],
        )
    else:
        processed = analyzer.calcuTheteAndKeyForGroupedSelection(
            fileName=target["protein"],
            chain=target["chain"],
            seq_values=target["seqnums"],
            chain_identities=target["ligands"],
            output_identifier=target["identifier"],
        )

    if not processed:
        return None

    return write_summary_row(target, analyzer)


def main(input_path, csv_file, lexical_file, output_path, summary_csv, results_dir, dtheta_value, dlen_value, num_labels_value, num_cores, mode):
    """Main entry point for both CLI usage and batch execution."""
    global CSV_FILE_PATH, CSV_FILE_LEXICAL_PATH, PDB_DIR_PATH, PROTEIN_DIR_PATH, SUMMARY_CSV_PATH, dtheta, dLen, numOfLabels

    if num_cores < 1:
        raise ValueError("--num_cores must be at least 1.")

    normalized_mode = normalize_mode(mode)
    resolved_results_dir = resolve_path(results_dir) if results_dir else None
    CSV_FILE_PATH = resolve_path(csv_file)
    CSV_FILE_LEXICAL_PATH = resolve_path(lexical_file)
    PDB_DIR_PATH = resolve_output_path(input_path, resolved_results_dir)
    PROTEIN_DIR_PATH = resolve_output_path(output_path, resolved_results_dir)
    SUMMARY_CSV_PATH = resolve_output_path(summary_csv, resolved_results_dir)
    dtheta = dtheta_value
    dLen = dlen_value
    numOfLabels = num_labels_value

    if resolved_results_dir:
        os.makedirs(resolved_results_dir, exist_ok=True)
    os.makedirs(PDB_DIR_PATH, exist_ok=True)
    os.makedirs(PROTEIN_DIR_PATH, exist_ok=True)
    ensure_parent_dir(SUMMARY_CSV_PATH)

    protein_list, processing_targets = collect_grouped_entries(CSV_FILE_PATH, normalized_mode)
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
        delayed(process_target)(target, lexical_map) for target in processing_targets
    )

    summary_rows = deduplicate_summary_rows([row for row in results if row is not None])
    write_summary_csv(SUMMARY_CSV_PATH, summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate carbohydrate key triplets using either explicit residue targets or grouped protein-chain targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--input_path", type=str, required=True, help="Directory where PDB files will be stored")
    parser.add_argument("-c", "--csv_file", type=str, required=True, help="Path to the carbohydrate target CSV file")
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
            f"Use '{MODE_ALL_4}' to preserve the original row-wise residue behavior, "
            f"or '{MODE_ID_CHAIN}' to merge all ligand names and sequence numbers by protein+chain."
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
    )
