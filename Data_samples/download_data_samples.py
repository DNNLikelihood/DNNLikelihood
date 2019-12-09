import os
import requests

def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)
        
def download_file(url, filename):
    req = requests.get(url)
    file = open(filename, 'wb')
    for chunk in req.iter_content(100000):
        file.write(chunk)
    file.close()
    print ("File "+filename+" of size "+file_size(filename)+" downloaded in "+os.getcwd())

file_list = ["DNNLikelihood_B1", "DNNLikelihood_B2",
             "DNNLikelihood_B3", "DNNLikelihood_F1", 
             "DNNLikelihood_F2", "DNNLikelihood_F3", 
             "likelihood_biased_gm_11M", "likelihood_mixed_sm_13_thinned1000_11M",
             "likelihood_unbiased_sm_13_thinned1000_11M", "pseudo_experiments_hybrid",
             "pseudo_experiments_profile"]

for file in file_list:
    download_file(
        "https://zenodo.org/record/3567822/files/"+file+".pickle?download=1",
        file+".pickle"
    )
