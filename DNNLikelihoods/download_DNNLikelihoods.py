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
    'https://cernbox.cern.ch/index.php/s/wHU4jI10rBp3Aj3/download',
    'F3_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/4bcAE42DHF8Iq7u/download',
    'F3_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/KsjuU49dRHTuNLc/download',
    'F2_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/or6QGHF859jkOws/download',
    'F2_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/08tBo9T7SxQQ5jA/download',
    'F1_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/Odom3T8v672rXbf/download',
    'F1_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/njyojL1wBRMi9D3/download',
    'B3_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/OlAUv5HxTveDG6Y/download',
    'B3_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/xdsaP9Mxbp5vhtU/download',
    'B2_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/ruqdvRW1ksEAgw3/download',
    'B2_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/08O2KGUYqlt0jm8/download',
    'B1_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/jAQcjB48osfkRhC/download',
    'B1_model.h5'
)









