import math
import pandas as pd
import requests
import sys
import csv
import multiprocessing
from joblib import Parallel, delayed
from utils.theta_utils import thetaClass
from utils.distance_utils import dist12Class

args= sys.argv[1:]
if len(args) < 3 :
    print("please specify all parameters")
    print("command : python key_triplets.py full_path csv_file_path csv_file_lexical_path pdf_folder_path protein_path")
    quit(0)

# Full path
Relative_PATH = args[0]
# csv_file_path contains protein and chain info
CSV_FILE_PATH = args[1]
# csv_file_lexical_path contains atoms and their code
CSV_FILE_LEXICAL_PATH = args[2]
# Download and store the pdb files in this directory (make sure to make this directory)
PDB_DIR_PATH = args[3]
# Calculate and store the keys and frequency in this directory (make sure to make this directory)
PROTEIN_DIR_PATH = args[4]

class AminoAcidAnalyzer:
    def __init__(self, dtheta, dLen, numOfLabels):
        self.dtheta = dtheta
        self.dLen = dLen
        self.numOfLabels = numOfLabels
        # These are the different labels of the atoms with their unique code (From CSV File)
        self.aminoAcidLabelWithCode = {}
        # Store animo acids code, x coordinate, y coordinate, z coordinate
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
        # totak number of keys, and max distance list
        self.totalKeys = []
        self.maxDistList = []
        # This is reading the csv file and generating proteins list containing fileName and chain
        self.proteinList = []
        # New dictionary for lexical code for specific chain only
        self.newDictForLexicalCode = {}
        # dictionary for storing seq and chain identity only
        self.seqchainIdentity = {}
        self.skip = False

    def readCSVProteinChain(self, csvFile):
        df = pd.read_csv(csvFile)
        self.proteinList = [f"{protein}_{chain}" for protein, chain in zip(df.protein, df.chain)]
        print("This is the list of protein with chain: ", self.proteinList) 
        # self.proteinList = ["5ZU1_E"]
    
    # Code to download the data set from rcsb.org
    def downloadDataSet(self, download_dir = Relative_PATH+PDB_DIR_PATH):
        for file in self.proteinList:
            pdb_url = 'https://files.rcsb.org/download/'+str(file.split("_")[0])+".pdb"  
            save_path = str(file.split("_")[0])+'.pdb'
            print(save_path)
            try:
                response = requests.get(pdb_url)
                if response.status_code == 200:
                    with open(download_dir+"/"+save_path, 'wb') as pdb_file:
                        pdb_file.write(response.content)
                    print("PDB file for" ,file," downloaded and saved as ",save_path,".")
                else:
                    print("Failed to download PDB file for", file ,". Status code:", response.status_code)

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
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2 + (z2 - z1) **2)

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
        self.aminoAcidLabelWithCode = dict(zip(df['atom'], df['seq']))
        self.aminoAcidLabelWithCode.update(dict(zip(df['ATOM'], df['seq'])))

    def readSeqAndIdentityChain(self, fileName, chain):
        # with open("drug/key_generator/pdb_files/"+fileName+".pdb", "r") as pdbFile:
        with open(Relative_PATH+"pdb_files/"+fileName+".pdb", "r") as pdbFile:
            for line in pdbFile:
                # Do not take CA if the peptide bond is broken i.e after the TER
                if ((line[0:6].rstrip()=="ENDMDL") or (line[0:6].rstrip()=='TER'and line[21].rstrip()==chain)):
                    break
                if (line[0:6].rstrip()=="MODEL" and int(line[10:14].rstrip())>1):
                    break                       
                if (line.startswith("HETATM") and line[21:22].strip()==chain):
                    self.seqchainIdentity[line[22:27].strip()] = line[17:20].strip()
        
    # This function is used to extract alpha carbons and then calculate theta and key
    def calcuTheteAndKey(self, fileName, chain, seq_value, chain_identity, protein_path= Relative_PATH+PROTEIN_DIR_PATH):
        # Resetting lists for each protein calculation
        self.totalKeys = []
        self.maxDistList = []
        self.keyFreq = {}
        incrementVal=0
        self.newDictForLexicalCode = {}
        self.aminoAcidCode = {}
        self.aminoSeqNum = {}
        # with open("drug/key_generator/pdb_files/"+fileName+".pdb", "r") as pdbFile:
        with open(Relative_PATH+"pdb_files/"+fileName+".pdb", "r") as pdbFile:
            for line in pdbFile:
                try: 
                    # Do not take CA if the peptide bond is broken i.e after the TER
                    if ((line[0:6].rstrip()=="ENDMDL") or (line[0:6].rstrip()=='TER'and line[21].rstrip()==chain)):
                        break
                    if (line[0:6].rstrip()=="MODEL" and int(line[10:14].rstrip())>1):
                        break                       
                    if (line.startswith("HETATM") and line[21:22].strip()==chain and line[22:27].strip() == seq_value and line[77:80].strip() != "H" and line[77:80].strip() != "D"):
                        
                        # Reading the lines in pdb file and then assigning residue (VAL) to its value (20)
                        self.aminoAcidCode[incrementVal]=int(self.aminoAcidLabelWithCode[line[13:16].rstrip()])
                        # This is the sequence number of the amino acid stored (Residue seq number)
                        self.aminoSeqNum[incrementVal]=str(line[22:27])
                        self.xCoordinate[incrementVal]=(float(line[30:38]))
                        self.yCoordinate[incrementVal]=(float(line[38:46]))
                        self.zCoordinate[incrementVal]=(float(line[46:54]))
                        self.newDictForLexicalCode[line[13:16].rstrip()] = int(self.aminoAcidLabelWithCode[line[13:16].rstrip()])
                        incrementVal+=1
                except Exception as e:
                    print("Their is an error in: ", line, pdbFile)
                    print(e)
        # print(f"The length of each {chain_identity}",len(self.aminoAcidCode))

        # if ((chain_identity == "G" and len(self.aminoAcidCode) == 23) or (chain_identity == "A" and len(self.aminoAcidCode) == 22) or (chain_identity == "C" and len(self.aminoAcidCode) == 20) or
        #     (chain_identity == "U" and len(self.aminoAcidCode) == 20) or (chain_identity == "DG" and len(self.aminoAcidCode) == 22) or (chain_identity == "DA" and len(self.aminoAcidCode) == 21) or
        #     (chain_identity == "DC" and len(self.aminoAcidCode) == 19) or (chain_identity == "DT" and len(self.aminoAcidCode) == 20)):
        #     self.skip = True
        tripletsFile = open(f"{protein_path}/{fileName}_{chain}_{seq_value}_{chain_identity}.keys_theta29_dist18", "w")
        keyFreqFile = open(f"{protein_path}/{fileName}_{chain}_{seq_value}_{chain_identity}.keys_Freq_theta29_dist18", "w")
        # This is the four rules that calculates the label, theta, and key (3 amino acids form a triplet)
        for i in range(0, len(self.aminoAcidCode) - 2):
            for j in range(i+1, len(self.aminoAcidCode) - 1):
                for k in range(j+1, len(self.aminoAcidCode)):
                    # This is a dictionary to keep the index and the labels
                    labelIndexToUse={}
                    # First, Second and Third label and Index
                    labelIndexToUse[self.aminoAcidCode[i]]=i
                    labelIndexToUse[self.aminoAcidCode[j]]=j
                    labelIndexToUse[self.aminoAcidCode[k]]=k
                    # First, Second and Third amino label list
                    self.initAminoLabel[0]=self.aminoAcidCode[i]
                    self.initAminoLabel[1]=self.aminoAcidCode[j]
                    self.initAminoLabel[2]=self.aminoAcidCode[k]
                    # Sorted labels from above list 
                    sortedAminoLabel=list(self.initAminoLabel)
                    # Reverse order from above sorted list
                    sortedAminoLabel.sort(reverse=True)

                    # The fourth case when l1=l2=l3
                    if (sortedAminoLabel[0] == sortedAminoLabel[1]) and (sortedAminoLabel[1]==sortedAminoLabel[2]):
                        distance1_2 = self.calDistance(i,j)
                        distance1_3 = self.calDistance(i,k)
                        distance2_3 = self.calDistance(j,k)
                        if distance1_2 >= (max(distance1_2,distance1_3,distance2_3)):
                            l1_index0=i
                            l2_index1=j
                            l3_index2=k
                        elif distance1_3 >= (max(distance1_2,distance1_3,distance2_3)):
                            l1_index0=i
                            l2_index1=k
                            l3_index2=j
                        else:
                            l1_index0=j
                            l2_index1=k
                            l3_index2=i

                    # Third condition when l1=l2>l3
                    elif(sortedAminoLabel[0]==sortedAminoLabel[1])and(sortedAminoLabel[1]!=sortedAminoLabel[2]):
                        l3_index2 = labelIndexToUse[sortedAminoLabel[2]]
                        indices = self.findTheIndex(l3_index2,i,j,k)
                        first = l3_index2
                        second = indices[0]
                        third  =indices[1]
                        distance1_3=self.calDistance(second,first)
                        distance2_3=self.calDistance(third,first)
                        if distance1_3>=distance2_3:
                            l1_index0=indices[0]
                            l2_index1=indices[1]	
                        else:
                            l1_index0=indices[1]
                            l2_index1=indices[0]

                    # Second condition when l1>l2=l3     
                    elif(sortedAminoLabel[0]!=sortedAminoLabel[1])and(sortedAminoLabel[1]==sortedAminoLabel[2]):
                        l1_index0=labelIndexToUse[sortedAminoLabel[0]]
                        indices = self.findTheIndex(l1_index0,i,j,k)
                        if self.calDistance(l1_index0,indices[0])>= self.calDistance(l1_index0,indices[1]):
                            l2_index1=indices[0]
                            l3_index2=indices[1]	
                        else:
                            l3_index2=indices[0]
                            l2_index1=indices[1]

                    # First condition when l1!=l2!=l3
                    elif(sortedAminoLabel[0]!=sortedAminoLabel[1])and(sortedAminoLabel[0]!=sortedAminoLabel[2])and(sortedAminoLabel[1]!=sortedAminoLabel[2]):
                        # Getting the index from the labelIndexToUse from sortedAminoLabel use
                        for index in range(0,3):
                            self.sortedAminoIndex[index]=labelIndexToUse[sortedAminoLabel[index]]
                        l1_index0=self.sortedAminoIndex[0]
                        l2_index1=self.sortedAminoIndex[1]
                        l3_index2=self.sortedAminoIndex[2]

                    distance01=self.calDistance(l1_index0,l2_index1)
                    # Calculating the mid distance
                    midDis01 = distance01/2
                    distance02=self.calDistance(l1_index0,l3_index2)
                    distance12=self.calDistance(l2_index1,l3_index2)
                    # Calculating the max distance (D)
                    maxDistance=max(distance01,distance02,distance12)
                    # Calculating the mid point 
                    m1 = (self.xCoordinate[l1_index0]+ self.xCoordinate[l2_index1])/2
                    m2 = (self.yCoordinate[l1_index0]+ self.yCoordinate[l2_index1])/2
                    m3 = (self.zCoordinate[l1_index0]+ self.zCoordinate[l2_index1])/2

                    # Calculating the d3 distance
                    d3 = math.sqrt((m1 - self.xCoordinate[l3_index2])**2+(m2 - self.yCoordinate[l3_index2])**2+(m3 - self.zCoordinate[l3_index2])**2)

                    # Calculating thetaAngle1
                    thetaAngle1 = 180*(math.acos((distance02**2-midDis01**2-d3**2)/(2*midDis01*d3)))/3.14

                    # Check in which category does the angle falls
                    if thetaAngle1<=90:
                        theta = thetaAngle1
                    else:
                        theta = abs(180-thetaAngle1)

                    # Calculating the bin values for theta and max distance
                    binTheta = self.thetaClass(theta)
                    binLength = self.dist12Class(maxDistance)
                    
                    aminoAcidR1 = list(self.newDictForLexicalCode.keys())[l1_index0]
                    aminoAcidR2 = list(self.newDictForLexicalCode.keys())[l2_index1]
                    aminoAcidR3 = list(self.newDictForLexicalCode.keys())[l3_index2]
                    # These are the sequence number of the three amino acids
                    seqNumber1 = list(self.aminoSeqNum.values())[l1_index0]
                    seqNumber2 = list(self.aminoSeqNum.values())[l2_index1]
                    seqNumber3 = list(self.aminoSeqNum.values())[l3_index2]

                    # These are the coordinates of the three amino acids
                    aminoAcidC10, aminoAcidC11, aminoAcidC12 = self.xCoordinate[l1_index0], self.yCoordinate[l1_index0], self.zCoordinate[l1_index0]
                    aminoAcidC20, aminoAcidC21, aminoAcidC22 = self.xCoordinate[l2_index1], self.yCoordinate[l2_index1], self.zCoordinate[l2_index1]
                    aminoAcidC30, aminoAcidC31, aminoAcidC32 = self.xCoordinate[l3_index2], self.yCoordinate[l3_index2], self.zCoordinate[l3_index2]

                    # Calculating the triplets key value
                    tripletKeys = dLen*dtheta*(numOfLabels**2)*(self.aminoAcidCode[l1_index0]-1)+dLen*dtheta*(numOfLabels)*(self.aminoAcidCode[l2_index1]-1)+dLen*dtheta*(self.aminoAcidCode[l3_index2]-1)+dtheta*(binLength-1)+(binTheta-1)

                    # Total number of keys and max distance list
                    self.totalKeys.append(tripletKeys)
                    self.maxDistList.append(maxDistance)

                    # Filtering out the distinct keys
                    if tripletKeys in self.keyFreq:
                        self.keyFreq[tripletKeys]+=1
                    else:
                        self.keyFreq[tripletKeys] = 1

                    # These are the info of all the triplets
                    tripletInfoAll = (str(tripletKeys)+"\t"+str(aminoAcidR1)+"\t"+str(seqNumber1)+"\t"+str(aminoAcidR2)+"\t"+str(seqNumber2)+"\t"+str(aminoAcidR3)+"\t"+str(seqNumber3)+"\t"+str(binTheta)+"\t"+str(theta)+"\t"+str(binLength)+"\t"+str(maxDistance)+"\t"+str(aminoAcidC10)+"\t"+str(aminoAcidC11)+"\t"+str(aminoAcidC12)+"\t"+str(aminoAcidC20)+"\t"+str(aminoAcidC21)+"\t"+str(aminoAcidC22)+"\t"+str(aminoAcidC30)+"\t"+str(aminoAcidC31)+"\t"+str(aminoAcidC32)+"\n")
                    tripletsFile.writelines(tripletInfoAll)

        # Storing the distinct keys in a file
        for values in self.keyFreq:
            keyFreqFile.writelines([str(values), '\t', str(self.keyFreq[values]), "\n"]) 

        # else:
        #     self.skip = False
# Usage
if __name__ == "__main__":
    dtheta = 29
    dLen = 18
    numOfLabels = 112
    analyzer = AminoAcidAnalyzer(dtheta, dLen, numOfLabels)
    # Generate list of proteins from csv file containing list of proteins and chain
    analyzer.readCSVProteinChain(Relative_PATH+CSV_FILE_PATH)
    analyzer.downloadDataSet()
    analyzer.readDrugLexicalCsv(Relative_PATH+CSV_FILE_LEXICAL_PATH)        
    numOfCores = multiprocessing.cpu_count()
    ##### This is to write in a csv file
    
    # csv_file_path = "drug/key_generator/" + "drug_csv/proteinNumKeysDist.csv"
    csv_file_path = Relative_PATH + "drug_csv/proteinNumKeysDist.csv"
    header = ["Protein", "#atoms", "#keys", "#keys_with_freq", "max_distance", "min_distance"]

    def calculate_data(fileName):
        # analyzer.skip = False
        fileChain = fileName.split("_")
        fileN = fileChain[0]
        chain = fileChain[-1]
        # print(fileN, chain)
        analyzer.readSeqAndIdentityChain(fileN, chain)
        rows = []
        for seq_value, chain_identity in analyzer.seqchainIdentity.items():
            analyzer.calcuTheteAndKey(fileN, chain, seq_value, chain_identity)
            # if analyzer.skip == True:
            totalAtoms = str(len(analyzer.aminoAcidCode))
            totalKeys = str(len(analyzer.totalKeys))
            keysWithFreq = str(len(analyzer.keyFreq))
            maxDistance = str(max(analyzer.maxDistList))
            minDistance = str(min(analyzer.maxDistList))
            row = [f"{fileN}_{chain}_{seq_value}_{chain_identity}", totalAtoms, totalKeys, keysWithFreq, maxDistance, minDistance]
            rows.append(row)
        return rows

    results = Parallel(n_jobs=numOfCores, verbose=50)(delayed(calculate_data)(fileName) for fileName in analyzer.proteinList)

    # Flattenning the list of lists
    flat_results = [item for sublist in results for item in sublist]

    # Converting the results to a DataFrame
    result_df = pd.DataFrame(flat_results, columns=header)

    # Removing duplicates from the DataFrame (safe case)
    result_df.drop_duplicates(inplace=True)

    # Writing the DataFrame to the CSV file
    result_df.to_csv(csv_file_path, index=False)

