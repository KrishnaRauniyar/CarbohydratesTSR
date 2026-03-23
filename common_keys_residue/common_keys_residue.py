import glob, os, argparse, time
import pandas as pd
from os.path import expanduser

parser = argparse.ArgumentParser(description='Common Keys')
parser.add_argument(
    '--path',
    '-path',
    metavar='path',
    help='Directory of input sample and other files.'
)

def common_keys(files):
    common_keys = []
    start = time.time()
    total_keys = {}

    for file in files:
        keys = {}
        with open(file, 'r') as f:
            for line in f:
                key, freq = line.split('\t')
                keys[key] = int(freq)

        total_keys[file] = keys

        if common_keys:
            common_keys = list(set(common_keys) & set(keys.keys()))
        else:
            common_keys = list(keys.keys())

    print('Time taken for Common Keys Calculation: ', (time.time() - start) / 60)
    return total_keys, common_keys


def extract_residue(filename):
    """
    Extract residue name from filename.
    Example: 7PNB_z_6_MAN.keys_theta29_dist18 → MAN
    """
    base = os.path.basename(filename)
    before = base.split(".keys")[0]
    residue = before.split("_")[-1]
    return residue


if __name__ == "__main__":
    args = parser.parse_args()

    # ----------------------------
    # Create results directory
    # ----------------------------
    result_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved in: {result_dir}\n")

    # ----------------------------
    # Load all key files
    # ----------------------------
    files = glob.glob(os.path.join(args.path, '*.keys_Freq_theta29_dist18*'))
    print("Total files found:", len(files))

    # ----------------------------
    # Auto-detect residues
    # ----------------------------
    residues = sorted(set(extract_residue(f) for f in files))
    print("Detected Residues:", residues)

    # ----------------------------
    # Process each residue
    # ----------------------------
    for residue in residues:
        print(f"\nProcessing Residue: {residue}")

        chain_files = [
            file for file in files
            if extract_residue(file) == residue
        ]

        if not chain_files:
            print(f"No files found for residue {residue}")
            continue

        total_keys, common_keys_list = common_keys(chain_files)
        print(f"Common Keys for {residue}: {len(common_keys_list)}")

        # Store results
        data = {
            'fileName': [],
            'total_keys': [],
            'common_keys': [],
            'sum_common_keys_freq': []
        }

        for file, keys in total_keys.items():
            common_count = len(common_keys_list)
            data['fileName'].append(os.path.splitext(os.path.basename(file))[0])
            data['total_keys'].append(len(keys))
            data['common_keys'].append(common_count)
            data['sum_common_keys_freq'].append(
                sum(keys[k] for k in common_keys_list if k in keys)
            )

        df = pd.DataFrame(data)

        # ------------------------------------------------
        # Save CSV into results directory
        # ------------------------------------------------
        out_csv = os.path.join(result_dir, f'common_keys_{residue}.csv')
        df.to_csv(out_csv, index=False)

        print(f"Saved: {out_csv}")
