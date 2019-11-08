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
    'https://cernbox.cern.ch/index.php/s/UBZT8CXm99s9G4r/download',
    'F3_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/oURF0Wk7NeU8USn/download',
    'F3_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/NUoeWaYxrBfA2nY/download',
    'F2_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/H9sPy3p7i7UWzOl/download',
    'F2_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/ytxdUMUXCimky8q/download',
    'F1_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/3emEYmtzYJOKO1H/download',
    'F1_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/s7eHfcskW1B3WEF/download',
    'B3_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/z5D1CZztiVjZeGR/download',
    'B3_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/aqTNBfvvqSrVq1c/download',
    'B2_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/KhiZVoIwjyxHge8/download',
    'B2_model.h5'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/fc7RN1bdVUxqTVA/download',
    'B1_model.onnx'
)

download_file(
    'https://cernbox.cern.ch/index.php/s/XKZzh4nr0LZKXPd/download',
    'B1_model.h5'
)









