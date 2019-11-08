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

    
download_file(
    'https://cernbox.cern.ch/index.php/s/kMDliPvcwARENeI/download',
    'pseudo_experiments_1000.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/ukTV5NiWbVFMcDJ/download',
    'likelihood_unbiased_sm_13_thinned1000_11M.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/yLDAFRcH6heeGUx/download',
    'likelihood_mixed_sm_13_thinned1000_11M.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/TGiq7Z987gSfgV4/download',
    'likelihood_biased_gm_11M.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/7uMinOngamBG1vu/download',
    'DNNLikelihood_F3.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/n9EqXg4h0NHDp5T/download',
    'DNNLikelihood_F2.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/Yfj9kuIjGCpFtsT/download',
    'DNNLikelihood_F1.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/Yfj9kuIjGCpFtsT/download',
    'DNNLikelihood_B3.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/woSYbbwNWzlcygQ/download',
    'DNNLikelihood_B2.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/nb85TAKf1ibt0rB/download',
    'DNNLikelihood_B1.pickle'
)









