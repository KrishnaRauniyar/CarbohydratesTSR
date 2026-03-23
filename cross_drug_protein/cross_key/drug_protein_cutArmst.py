import os
import csv
import numpy as np
import argparse
import pandas as pd
import requests
from joblib import Parallel, delayed

# Protein residues (static)
protein_residues = {
    'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP',
    'SER', 'THR', 'MET', 'CYS', 'HIS', 'LYS', 'ARG', 'ASP', 'GLU',
    'ASN', 'GLN', 'TPO', 'SEP', 'PTR'
}

# Reading the CSV file and extract protein names and chains
def read_csv_protein_chain(csv_file):
    df = pd.read_csv(csv_file)
    protein_list = set([protein for protein in df.protein])
    print("This is the length and list of protein with chain: ", len(protein_list), protein_list)
    return protein_list

# Read drug residues from carb CSV (3rd column = carb_name) For the drug/carb residues
def read_drug_residues_from_carb_csv(carb_csv_file):
    df = pd.read_csv(carb_csv_file)

    if "carb_name" not in df.columns:
        raise ValueError(
            f"Expected column 'carb_name' in {carb_csv_file}, but found: {list(df.columns)}"
        )

    # unique, non-null, stripped
    residues = (
        df["carb_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    drug_residues = set(residues.tolist())
    print(f"Loaded {len(drug_residues)} drug residues from carb CSV: {sorted(drug_residues)}")
    return drug_residues

# Reading the lexical file to create a dictionary for sequence numbers for each atoms
def read_lexical_file(lexical_file):
    df = pd.read_csv(lexical_file)
    lexical_dict = dict(zip(df['atom'], df['seq']))
    # keep your original behavior for 'ATOM' too (if present)
    if "ATOM" in df.columns:
        lexical_dict.update(dict(zip(df['ATOM'], df['seq'])))
    return lexical_dict

# Downloading the PDB files from rcsb.org
def download_pdb(file, download_dir):
    pdb_id = file.split("_")[0]
    pdb_url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    save_path = os.path.join(download_dir, f"{pdb_id}.pdb")
    try:
        response = requests.get(pdb_url, timeout=60)
        if response.status_code == 200:
            with open(save_path, 'wb') as pdb_file:
                pdb_file.write(response.content)
            print(f"PDB file for {file} downloaded and saved as {save_path}.")
        else:
            print(f"Failed to download PDB file for {file}. Status code: {response.status_code}")

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

# Parsing the PDB file
def parse_pdb(file_path):
    atoms = []
    with open(file_path, 'r') as f:
        for line in f:
            if (line.startswith("ATOM") or line.startswith("HETATM")) and line[77:80].strip() not in ("H", "D"):
                atom = {
                    'name': line[12:16].strip(),
                    'residue': line[17:20].strip(),
                    'chain': line[21].strip(),
                    'residue_number': line[22:26].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54])
                }
                atoms.append(atom)
    return atoms

# Calculating the distance between two atoms
def calculate_distance(atom1, atom2):
    return np.linalg.norm(
        np.array([atom1['x'], atom1['y'], atom1['z']]) -
        np.array([atom2['x'], atom2['y'], atom2['z']])
    )

def process_pdb_file(file_path, name_of_file, lexical_dict, drug_residues, cross_key_cut):
    atoms = parse_pdb(file_path)
    drug_atoms = [atom for atom in atoms if atom['residue'] in drug_residues]
    protein_atoms = [atom for atom in atoms if atom['residue'] in protein_residues]

    results = []
    for drug_atom in drug_atoms:
        for protein_atom in protein_atoms:
            distance = calculate_distance(drug_atom, protein_atom)
            if distance <= cross_key_cut:
                results.append([
                    f"{name_of_file}_{drug_atom['chain']}_{drug_atom['residue']}_{drug_atom['residue_number']}_{drug_atom['name']}",
                    lexical_dict.get(drug_atom['name'], "N/A"),
                    f"{drug_atom['x']:.3f}_{drug_atom['y']:.3f}_{drug_atom['z']:.3f}",
                    f"{name_of_file}_{protein_atom['chain']}_{protein_atom['residue']}_{protein_atom['residue_number']}_{protein_atom['name']}",
                    lexical_dict.get(protein_atom['name'], "N/A"),
                    f"{protein_atom['x']:.3f}_{protein_atom['y']:.3f}_{protein_atom['z']:.3f}",
                    f"{distance:.1f}"
                ])
    return results


def main(directory_path, csv_file, lexical_file, carb_csv_file, output_csv, cross_key_cut):
    protein_list = read_csv_protein_chain(csv_file)
    lexical_dict = read_lexical_file(lexical_file)
    drug_residues = read_drug_residues_from_carb_csv(carb_csv_file)

    # Downloading the PDB files in parallel (-1 means use all the cores available)
    Parallel(n_jobs=-1, verbose=5)(
        delayed(download_pdb)(file, directory_path) for file in protein_list
    )

    # Processing the PDB files in parallel
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_pdb_file)(
            os.path.join(directory_path, f"{filename.split('.')[0]}.pdb"),
            filename.split('.')[0],
            lexical_dict,
            drug_residues,
            cross_key_cut
        )
        for filename in os.listdir(directory_path) if filename.endswith(".pdb")
    )

    # Writing results to the CSV file
    with open(output_csv, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([
            'drug', 'drug_seq', 'drug_coordinates',
            'protein', 'protein_seq', 'protein_coordinates',
            f'distance(angstrom)<=cut({cross_key_cut})'
        ])
        for result in results:
            csv_writer.writerows(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug-protein cross key calculation with configurable cutoff")
    parser.add_argument('-p', '--input_path', type=str, required=True, help="Path to download PDB files")
    parser.add_argument('-c', '--csv_file', type=str, required=True, help="Path to the input CSV file (protein list)")
    parser.add_argument('-l', '--lexical_file', type=str, required=True, help="Path to the lexical CSV file")
    parser.add_argument('--carb_csv', type=str, required=True, help="CSV containing carb_name column (drug residues)")
    parser.add_argument('-o', '--output_csv', type=str, required=True, help="Path to the output CSV file")

    # cutoff argument
    parser.add_argument('--cross_key_cut', type=float, default=3.0,
                        help="Distance cutoff (angstrom) for cross keys (default: 3.0)")

    args = parser.parse_args()
    main(args.input_path, args.csv_file, args.lexical_file, args.carb_csv, args.output_csv, args.cross_key_cut)
