"""
Generate carbohydrate atom-triplet keys from a CSV of explicit residue targets.

This script reads rows containing `protein`, `chain`, `carb_name`, and `carb_id`,
downloads the required PDB files when needed, selects each requested carbohydrate
residue, and reuses the existing triplet/theta/distance key-calculation logic to
write per-residue key files plus a summary CSV.
"""

import argparse
import math
import os
import pandas as pd
import requests
import sys
import csv
import multiprocessing
from joblib import Parallel, delayed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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


class AminoAcidAnalyzer:
    def __init__(self, dtheta, dLen, numOfLabels):
        self.dtheta = dtheta
        self.dLen = dLen
        self.numOfLabels = numOfLabels
        # These are the different labels of the atoms with their unique code (From CSV File)
        self.aminoAcidLabelWithCode = {}
        # Store amino acids code, x coordinate, y coordinate, z coordinate
        self.aminoAcidCode = {}
        self.xCoordinate = {}
        self.yCoordinate = {}
        self.zCoordinate = {}
        # This is to store the sequence number
        self.aminoSeqNum = {}
        # These three holds the label code of three amino acid, sorted them, and then store the sorted index
        self.initAminoLabel = [0, 0, 0]
        self.sortedAminoLabel = [0, 0, 0]
        self.sortedAminoIndex = [0, 0, 0]
        # keys with its frequency
        self.keyFreq = {}
        # total number of keys, and max distance list
        self.totalKeys = []
        self.maxDistList = []
        # This is reading the csv file and generating proteins list containing fileName and chain
        self.proteinList = []
        self.targetResidues = []
        # Preserve atom labels by index so alternate-location records do not collapse duplicate atom names.
        self.atomLabelByIndex = {}
        self.skip = False

    def _clean_csv_value(self, value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _parse_carb_ids(self, carb_id_value, row_number):
        # Split carb_id on ';', strip whitespace, and ignore empty values.
        carb_ids = []
        raw_value = self._clean_csv_value(carb_id_value)
        if not raw_value:
            print(f"Warning: skipping row {row_number} because carb_id is missing.")
            return carb_ids

        for seq_value in raw_value.split(";"):
            cleaned_value = seq_value.strip()
            if cleaned_value:
                carb_ids.append(cleaned_value)

        if not carb_ids:
            print(f"Warning: skipping row {row_number} because carb_id '{raw_value}' does not contain a valid residue sequence number.")

        return carb_ids

    def readCSVProteinTargets(self, csvFile):
        df = pd.read_csv(csvFile)
        required_columns = {"protein", "chain", "carb_name", "carb_id"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            print("Missing required columns in input CSV:", ", ".join(sorted(missing_columns)))
            quit(0)

        protein_chain_seen = set()
        for row_index, row in df.iterrows():
            # Parse the new CSV row by row so each protein/chain/carb_name/carb_id target can be processed explicitly.
            protein = self._clean_csv_value(row.get("protein"))
            chain = self._clean_csv_value(row.get("chain"))
            carb_name = self._clean_csv_value(row.get("carb_name"))
            carb_ids = self._parse_carb_ids(row.get("carb_id"), row_index + 2)

            if not protein or not chain or not carb_name:
                print(
                    f"Warning: skipping row {row_index + 2} because protein='{protein}', chain='{chain}', or carb_name='{carb_name}' is missing."
                )
                continue

            if not carb_ids:
                continue

            protein_chain = f"{protein}_{chain}"
            if protein_chain not in protein_chain_seen:
                protein_chain_seen.add(protein_chain)
                self.proteinList.append(protein_chain)

            for seq_value in carb_ids:
                self.targetResidues.append(
                    {
                        "protein": protein,
                        "chain": chain,
                        "carb_name": carb_name,
                        "seq_value": seq_value,
                    }
                )

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

    # Calculating the distance between the amino acids
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

    def readDrugLexicalCsv(self, csvFile):
        df = pd.read_csv(csvFile)
        self.aminoAcidLabelWithCode = dict(zip(df["atom"], df["seq"]))
        self.aminoAcidLabelWithCode.update(dict(zip(df["ATOM"], df["seq"])))

    # This function is used to extract atoms for a specific residue and then calculate theta and key.
    def calcuTheteAndKey(self, fileName, chain, seq_value, chain_identity, protein_path=None):
        if protein_path is None:
            protein_path = PROTEIN_DIR_PATH
        # Resetting lists for each protein calculation
        self.totalKeys = []
        self.maxDistList = []
        self.keyFreq = {}
        incrementVal = 0
        self.atomLabelByIndex = {}
        self.aminoAcidCode = {}
        self.aminoSeqNum = {}
        # Original key-calculation logic below is reused unchanged; only the residue-selection filter now includes chain_identity and seq_value from the new CSV.
        with open(os.path.join(PDB_DIR_PATH, fileName + ".pdb"), "r") as pdbFile:
            for line in pdbFile:
                try:
                    # Keep scanning after TER because carbohydrate HETATM records can appear later in the same chain.
                    if line[0:6].rstrip() == "ENDMDL":
                        break
                    if (line[0:6].rstrip() == "MODEL" and int(line[10:14].rstrip()) > 1):
                        break
                    if (
                        line.startswith("HETATM")
                        and line[16:17].strip() in ("", "A") # accept blank or 'A' alternate location indicator for carbohydrate residues
                        and line[21:22].strip() == chain
                        and line[17:20].strip() == chain_identity
                        and line[22:27].strip() == seq_value
                        and line[77:80].strip() != "H"
                        and line[77:80].strip() != "D"
                    ):

                        # Reading the lines in pdb file and then assigning residue atom to its lexical value.
                        self.aminoAcidCode[incrementVal] = int(self.aminoAcidLabelWithCode[line[13:16].rstrip()])
                        # This is the sequence number of the amino acid stored (Residue seq number)
                        self.aminoSeqNum[incrementVal] = str(line[22:27])
                        self.xCoordinate[incrementVal] = float(line[30:38])
                        self.yCoordinate[incrementVal] = float(line[38:46])
                        self.zCoordinate[incrementVal] = float(line[46:54])
                        self.atomLabelByIndex[incrementVal] = line[13:16].rstrip()
                        incrementVal += 1
                except Exception as e:
                    print("Their is an error in: ", line, pdbFile)
                    print(e)

        if len(self.aminoAcidCode) < 3:
            print(
                f"Warning: skipping {fileName}_{chain}_{seq_value}_{chain_identity} because only {len(self.aminoAcidCode)} matching non-hydrogen atoms were found."
            )
            return False

        tripletsFile = open(os.path.join(protein_path, f"{fileName}_{chain}_{seq_value}_{chain_identity}.keys_theta29_dist18"), "w")
        keyFreqFile = open(os.path.join(protein_path, f"{fileName}_{chain}_{seq_value}_{chain_identity}.keys_Freq_theta29_dist18"), "w")
        # This is the four rules that calculates the label, theta, and key (3 amino acids form a triplet)
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

        tripletsFile.close()
        keyFreqFile.close()
        return True

def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(os.getcwd(), path_value))


def resolve_output_path(path_value, results_dir=None):
    if os.path.isabs(path_value):
        return path_value
    if results_dir:
        return os.path.abspath(os.path.join(results_dir, path_value))
    return resolve_path(path_value)


def ensure_parent_dir(path_value):
    directory = os.path.dirname(path_value)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main(input_path, csv_file, lexical_file, output_path, summary_csv, results_dir, dtheta_value, dlen_value, num_labels_value, num_cores):
    global CSV_FILE_PATH, CSV_FILE_LEXICAL_PATH, PDB_DIR_PATH, PROTEIN_DIR_PATH, SUMMARY_CSV_PATH, dtheta, dLen, numOfLabels

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

    analyzer = AminoAcidAnalyzer(dtheta, dLen, numOfLabels)
    # Generate the explicit residue targets from the new CSV input.
    analyzer.readCSVProteinTargets(CSV_FILE_PATH)
    analyzer.downloadDataSet()
    analyzer.readDrugLexicalCsv(CSV_FILE_LEXICAL_PATH)
    ##### This is to write in a csv file

    header = ["Protein", "#atoms", "#keys", "#keys_with_freq", "max_distance", "min_distance"]

    def calculate_data(target):
        fileN = target["protein"]
        chain = target["chain"]
        chain_identity = target["carb_name"]
        seq_value = target["seq_value"]

        processed = analyzer.calcuTheteAndKey(fileN, chain, seq_value, chain_identity)
        if not processed:
            return None

        totalAtoms = str(len(analyzer.aminoAcidCode))
        totalKeys = str(len(analyzer.totalKeys))
        keysWithFreq = str(len(analyzer.keyFreq))
        maxDistance = str(max(analyzer.maxDistList))
        minDistance = str(min(analyzer.maxDistList))
        row = [f"{fileN}_{chain}_{seq_value}_{chain_identity}", totalAtoms, totalKeys, keysWithFreq, maxDistance, minDistance]
        return row

    results = Parallel(n_jobs=num_cores, verbose=50)(delayed(calculate_data)(target) for target in analyzer.targetResidues)

    # Remove skipped rows before converting to a DataFrame.
    flat_results = [item for item in results if item is not None]

    # Converting the results to a DataFrame
    result_df = pd.DataFrame(flat_results, columns=header)

    # Removing duplicates from the DataFrame (safe case)
    result_df.drop_duplicates(inplace=True)

    # Writing the DataFrame to the CSV file
    result_df.to_csv(SUMMARY_CSV_PATH, index=False)


# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate carbohydrate key triplets using explicit protein/chain/carb_name/carb_id targets")
    parser.add_argument("-p", "--input_path", type=str, required=True, help="Directory where PDB files will be stored")
    parser.add_argument("-c", "--csv_file", type=str, required=True, help="Path to the carbohydrate target CSV file")
    parser.add_argument("-l", "--lexical_file", type=str, required=True, help="Path to the lexical CSV file")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Directory where key files will be written")
    parser.add_argument(
        "-s",
        "--summary_csv",
        type=str,
        default="proteinNumKeysDist.csv",
        help="Path to the summary CSV file (default: proteinNumKeysDist.csv; stored under results_dir when provided)",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=str,
        default=None,
        help="Optional base directory under which pdb_files, proteins, and summary_csv will be created",
    )
    parser.add_argument("--dtheta", type=int, default=29, help="Theta bin count parameter (default: 29)")
    parser.add_argument("--dlen", type=int, default=18, help="Distance bin count parameter (default: 18)")
    parser.add_argument("--num_labels", type=int, default=112, help="Number of lexical labels used in key generation (default: 112)")
    parser.add_argument(
        "--num_cores",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPU cores used by joblib Parallel (default: multiprocessing.cpu_count())",
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
    )
