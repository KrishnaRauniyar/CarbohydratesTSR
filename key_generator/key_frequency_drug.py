import os
import datetime
import argparse
import csv

def protein_freq_table(directory, relative_path, header):
    protein_data = {}
    all_keys = set()
    # Loop through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.keys_Freq_theta29_dist18'):  # Assuming the files have a .txt extension
            protein_name = filename.split('.')[0]  # Extract the protein name from the filename
            file_path = os.path.join(directory, filename)
            # Read each file and extract keys with frequencies
            with open(file_path, 'r') as file:
                keys_with_freq = {}
                for line in file:
                    key, freq = line.strip().split('\t')
                    keys_with_freq[key] = freq
                    all_keys.add(key)
                # Store the keys with frequencies for each protein name
                protein_data[protein_name] = keys_with_freq

    if header == 'yes':
        # This  will be the input file for the dnn
        with open(relative_path+'drug_csv/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0_header.csv', 'w', newline='') as csvfile:
            # Un comment this for header code
            proteinName = ['Protein Name'] + list(all_keys)
            writer = csv.DictWriter(csvfile, proteinName)
            writer.writeheader()
            for protein_name, data in protein_data.items():
            # With header code
                row_data = {'Protein Name': protein_name}
                row_data.update({key: data.get(key, 0) for key in all_keys})
                writer.writerow(row_data)

    else:
        # This  will be the input file for the feng code
        with open(relative_path+'drug_csv/localFeatureVect_theta29_dist18_NoFeatureSelection_keyCombine0.csv', 'w', newline='') as csvfile:
            for protein_name, data in protein_data.items():
            # Without header code
                protein_name = protein_name + ";"
                csvfile.write(protein_name + ','.join([str(data.get(key, 0)) for key in all_keys]) + '\n')

def main(opt_dict):
    protein_freq_table(directory= opt_dict['relative_path']+opt_dict['proteins_path'], relative_path= opt_dict['relative_path'], header= opt_dict['header_bool'])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute the protein data frequency")
    parser.add_argument('-r','--relative_path', type=str,required=True, help="relative path")
    parser.add_argument('-p','--proteins_path', type=str,required=True, help="protein_dir path")
    parser.add_argument('-H','--header_bool', type=str,required=True, choices=['yes', 'no'], help="header or not header")
    args = parser.parse_args()
    opt_dict = vars(args)
    start_time = datetime.datetime.now()
    print("Starting Time : ",start_time)
    main(opt_dict)
    end_time = datetime.datetime.now()
    print("Execution time", end_time - start_time)