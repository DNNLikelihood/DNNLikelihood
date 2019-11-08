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
    'https://cernbox.cern.ch/index.php/s/vyhK4gqydxQd8cA/download',
    'pseudo_experiments_1000.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/ZnUjrfRWP7Szsxa/download',
    'likelihood_unbiased_sm_13_thinned1000_11M.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/nUFpwKyPJcv1N1R/download',
    'likelihood_mixed_sm_13_thinned1000_11M.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/TWY9vgF0JUdYYrz/download',
    'likelihood_biased_gm_11M.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/dNdlGbWSgd5rY2v/download',
    'DNNLikelihood_F3.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/hiWaikQVyWWJap0/download',
    'DNNLikelihood_F2.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/qpypdBpD7lszN9a/download',
    'DNNLikelihood_F1.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/fG98ETHgn7vc94n/download',
    'DNNLikelihood_B4.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/jgTEY7eU3wGddvV/download',
    'DNNLikelihood_B3.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/EzbMUEU8CtXr8ge/download',
    'DNNLikelihood_B2.pickle'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/iUtbYo6rRFLyHMH/download',
    'DNNLikelihood_B1.pickle'
)









