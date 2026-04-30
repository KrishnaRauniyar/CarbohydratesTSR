import datetime
import argparse
import csv
from pathlib import Path


HEADER_OUTPUT_FILE = "localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv"
NO_HEADER_OUTPUT_FILE = "localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0.csv"
KEY_FREQ_SUFFIX = ".keys_Freq_theta29_dist18"


def protein_freq_table(input_dir, output_dir, header):
    protein_data = {}
    all_keys = set()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for file_path in sorted(input_dir.iterdir()):
        if file_path.name.endswith(KEY_FREQ_SUFFIX):
            protein_name = file_path.name.split('.')[0]
            keys_with_freq = {}

            with file_path.open("r") as file:
                for line in file:
                    key, freq = line.strip().split('\t')
                    keys_with_freq[key] = freq
                    all_keys.add(key)

            protein_data[protein_name] = keys_with_freq

    output_dir.mkdir(parents=True, exist_ok=True)
    all_keys = sorted(all_keys)

    if header == 'yes':
        # This  will be the input file for the dnn
        output_csv = output_dir / HEADER_OUTPUT_FILE
        with output_csv.open('w', newline='') as csvfile:
            # Un comment this for header code
            proteinName = ['Protein Name'] + all_keys
            writer = csv.DictWriter(csvfile, proteinName)
            writer.writeheader()
            for protein_name, data in protein_data.items():
            # With header code
                row_data = {'Protein Name': protein_name}
                row_data.update({key: data.get(key, 0) for key in all_keys})
                writer.writerow(row_data)

    else:
        # This  will be the input file for the feng code
        output_csv = output_dir / NO_HEADER_OUTPUT_FILE
        with output_csv.open('w', newline='') as csvfile:
            for protein_name, data in protein_data.items():
            # Without header code
                protein_name = protein_name + ";"
                csvfile.write(protein_name + ','.join([str(data.get(key, 0)) for key in all_keys]) + '\n')


def resolve_input_dir(args):
    if args.input_dir:
        return Path(args.input_dir)

    if args.relative_path and args.proteins_path:
        return Path(args.relative_path) / args.proteins_path

    raise ValueError(
        "Provide --input-dir, or provide both --relative_path and --proteins_path."
    )


def resolve_output_dir(args, input_dir):
    if args.output_dir:
        return Path(args.output_dir)

    return Path(input_dir).parent


def main(opt_dict):
    input_dir = resolve_input_dir(opt_dict)
    output_dir = resolve_output_dir(opt_dict, input_dir)
    protein_freq_table(input_dir=input_dir, output_dir=output_dir, header=opt_dict.header_bool)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute the protein data frequency")
    parser.add_argument('-i','--input-dir', type=str, help="directory containing *.keys_Freq_theta29_dist18 files")
    parser.add_argument('-o','--output-dir', type=str, help="directory where the localFeatureVect CSV will be written")
    parser.add_argument('-r','--relative_path', type=str, help="base path used with --proteins_path for backward compatibility")
    parser.add_argument('-p','--proteins_path', type=str, help="protein directory path used with --relative_path for backward compatibility")
    parser.add_argument('-H','--header_bool', type=str,required=True, choices=['yes', 'no'], help="header or not header")
    args = parser.parse_args()
    start_time = datetime.datetime.now()
    print("Starting Time : ",start_time)
    main(args)
    end_time = datetime.datetime.now()
    print("Execution time", end_time - start_time)
