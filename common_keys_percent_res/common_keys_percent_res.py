import glob
import os
import argparse
import time
import pandas as pd
from os.path import expanduser

parser = argparse.ArgumentParser(description='Common Keys Percent based on Residue')
parser.add_argument(
    '--path',
    '-path',
    metavar='path',
    help='Directory of input sample and other files.'
)

def extract_residue(filename):
    """
    Extract residue name from filename.
    Example: 7PNB_z_6_MAN.keys_Freq_theta29_dist18 → MAN
    """
    base = os.path.basename(filename)
    before = base.split(".keys")[0]
    residue = before.split("_")[-1]
    return residue


def get_keys_percents(files, residue, result_dir):
    start = time.time()
    key_occurrence = {}

    # Count key occurrences across all files
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                key = line.split('\t')[0]
                key_occurrence[key] = key_occurrence.get(key, 0) + 1

    df = pd.DataFrame(key_occurrence.items(), columns=['key', 'key_occurrence'])
    df['total_files'] = len(files)
    df['key_percent'] = ((df['key_occurrence'] / len(files)) * 100).round(3)


    # Save output
    out_csv = os.path.join(result_dir, f'common_keys_percent_{residue}.csv')
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Time taken for Common Keys {residue} Percentage Calculation: {(time.time() - start) / 60} minutes")


if __name__ == "__main__":
    args = parser.parse_args()

    # ---------------------------------------
    # Create result directory
    # ---------------------------------------
    result_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(result_dir, exist_ok=True)
    print(f"Percentage results will be saved in: {result_dir}\n")

    # Load all key files
    files = glob.glob(os.path.join(args.path, '*.keys_Freq_theta29_dist18*'))
    print("Total files found:", len(files))

    # Auto-detect residues
    residues = sorted(set(extract_residue(f) for f in files))
    print("Detected Residues:", residues)

    # Compute percentages for each residue
    for residue in residues:
        print(f"\nProcessing Residue: {residue}")

        residue_files = [f for f in files if extract_residue(f) == residue]

        if not residue_files:
            print(f"No files found for residue {residue}")
            continue

        get_keys_percents(residue_files, residue, result_dir)
