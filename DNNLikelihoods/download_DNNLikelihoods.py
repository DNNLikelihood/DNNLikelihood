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


file_list = ["B1_model", "B2_model",
             "B3_model", "F1_model"
             "F2_model", "F3_model"]

for extension in [".h5", ".onnx"]:
    for file in file_list:
        download_file(
            "https://sandbox.zenodo.org/record/429558/files/"+file+extension+"?download=1",
            file+extension
        )
