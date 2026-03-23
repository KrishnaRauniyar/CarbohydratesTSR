import glob
import os
import argparse
import time
import pandas as pd
from os.path import expanduser

parser = argparse.ArgumentParser(description='Common Keys')
parser.add_argument(
    '--path',
    '-path',
    metavar='path',
    default=os.path.join(expanduser('~'), 'Research', 'Protien_Database',
                         'extracted_new_samples', 'testing'),
    help='Directory of input sample and other files.'
)

def extract_residue(filename):
    """
    Extract residue name (last underscore part before .keys)
    Example: 7PNB_z_6_MAN.keys_Freq_theta29_dist18 → MAN
    """
    base = os.path.basename(filename)
    before = base.split(".keys")[0]
    return before.split("_")[-1]


def get_keys_percents(files, residue):
    """
    Calculate % occurrence of each key for the given residue.
    """
    common_keys = {}

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                key = line.split('\t')[0]
                common_keys[key] = common_keys.get(key, 0) + 1

    total_files = len(files)
    key_percent = {key: round((count / total_files) * 100, 3)
                   for key, count in common_keys.items()}

    df = pd.DataFrame(list(key_percent.items()), columns=['key', f'{residue}_key%'])
    return df


if __name__ == "__main__":
    start = time.time()
    args = parser.parse_args()

    # ------------------------------------------------
    # Prepare result directory
    # ------------------------------------------------
    result_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(result_dir, exist_ok=True)
    print(f"Saving final table to: {result_dir}\n")

    # Load all key files
    files = glob.glob(os.path.join(args.path, "*.keys_Freq_theta29_dist18*"))
    print("Total files found:", len(files))

    # Auto-detect residues
    residues = sorted(set(extract_residue(f) for f in files))
    print("Detected Residues:", residues)

    # ------------------------------------------------
    # Build combined dataframe
    # ------------------------------------------------
    key_df = pd.DataFrame(columns=['key'])
    all_residue_dfs = []

    for residue in residues:
        print(f"Processing residue: {residue}")

        residue_files = [f for f in files if extract_residue(f) == residue]

        if len(residue_files) == 0:
            print(f"No files for residue {residue}, skipping.")
            continue

        df_residue = get_keys_percents(residue_files, residue)
        all_residue_dfs.append(df_residue)

        # Merge unique keys
        key_df = pd.merge(key_df, df_residue[['key']], on='key', how='outer')

    # Merge all percentage columns into final table
    final_df = key_df
    for df in all_residue_dfs:
        final_df = pd.merge(final_df, df, on='key', how='left')

    # Replace NaN with 0
    final_df.fillna(0, inplace=True)

    # Save output
    out_csv = os.path.join(result_dir, "common_keys_percent_one_vs_all.csv")
    final_df.to_csv(out_csv, index=False)

    print(f"Saved combined percentage file: {out_csv}")
    print(f"Total time taken: {(time.time() - start) / 60} minutes")
