import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
from bs4 import BeautifulSoup
import argparse

def parse_drugbank_xml(file_name: str, 
                       atc_codes: list = None, 
                       mp: int = None) -> pd.DataFrame:
    '''
    Parse the drug codes and info from the large XML file 
    downloaded from DrugBank.
    
    Inputs:
        file_name: str, path to the drugbank xml file
        
    Outputs:
        drugbank_df: pd.DataFrame, data extracted from drugbank xml file
    '''
    
    # open xml file
    print("Opening XML file...")
    with open(file_name, "r") as file:
        
        # convert xml to soup
        print("Creating XML file soup...")
        soup = BeautifulSoup(file.read(), "xml")
        
        # extract soup for each drug into a list
        print("Extracting all drug-related XML...")
        drugs = soup.find_all("drug")
     
    def parse_drug_info(drug_soup, atc_codes=None):
        '''
        Parse the drug codes and info for one drug.

        Inputs:
            drug_soup: BeautifulSoup, soup for one drug
            atc_code_list: str, list of atc codes in SIDER

        Outputs:
            drugbank_data: pd.DataFrame, data extracted from drugbank xml file
        '''

        # make sure soup is main drug info
        if drug_soup.find("atc-codes"):

            # extract ATC code
            atc_code = drug_soup.find("atc-codes").find("atc-code").get("code")

            # if drug is in SIDER data then find other data and add to record
            if atc_code in atc_codes:

                # extract name
                name = drug_soup.find("name").text.lower()

                # extract DrugBank ID
                db_id = drug_soup.find("drugbank-id", {"primary":"true"}).text

                # combine data in record
                record = {"drug_name": name,
                          "drugbank_id": db_id,
                          "drug_atc_code": atc_code}

                # add to full list of drugs
                return record               
    
    all_drugs = []
    print(f"Extracting info for all {len(drugs)} drugs...")
    if mp:
        print(f"Multiprocessing with {mp} workers...")
        pool = ThreadPool(mp)
        arguments = zip(drugs,
                        repeat(atc_codes))
        output = pool.starmap(parse_drug_info, arguments)
        all_drugs = [record for record in output if isinstance(record,dict)]
        pool.close()
        pool.join()
        
    else:
        print("Looping over drugs...")
        for drug in drugs:
            output = parse_drug_info(drug, atc_codes)
            if output:
                all_drugs.append(output)
                
    # create dataframe
    drugbank_data = pd.DataFrame(all_drugs)
    
    return drugbank_data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", 
                        type=str, 
                        help="path to input xml file")
    parser.add_argument("--atc_code_file", 
                        type=str, 
                        help="path to dataframe with ATC codes for filtering")
    parser.add_argument("--mp", 
                        type=int, 
                        help="number of multiprocessing workers", 
                        default=None)
    parser.add_argument("--save_path", 
                        type=str, 
                        help="save path for the output data (default=None, if no save name then don't save and print instead)", 
                        default=None)
    
    args = parser.parse_args()
    
    sider_data = pd.read_csv(args.atc_code_file)
    atc_codes = list(sider_data.drug_atc_code)

    data = parse_drugbank_xml(file_name = args.input_file, 
                              atc_codes = atc_codes, 
                              mp = args.mp)
    
    if args.save_path:
        data.to_csv(args.save_path)
        print("Data is saved!")
        print(data.head())
    else:
        print(data)