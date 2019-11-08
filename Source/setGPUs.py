import os
import subprocess
import numpy as np

if os.name == 'nt':
    try:
        os.environ['PATH'] += os.pathsep + r'C:\Program Files\NVIDIA Corporation\NVSMI'
        availableGPUs = (str(subprocess.check_output(["nvidia-smi", "-L"])).replace("\\n'","").replace("b'","").split("\\n"))
    except:
        print("nvidia-smi.exe not found it its system folder 'C:\\Program Files\\NVIDIA Corporation\\NVSMI'. Please modify the PATH accordingly.")
        availableGPUs = []
else:
    availableGPUs = (str(subprocess.check_output(["nvidia-smi", "-L"])).replace("\\n'","").replace("b'","").split("\\n"))
print(str(len(availableGPUs))+" GPUs available in current environment")
if len(availableGPUs) >0:
    print(availableGPUs)

def setGPUs(n):
    if len(availableGPUs)==0:
        print('No available GPUs in current environment.')
        return list()
    if np.amax(np.array(n)) > len(availableGPUs)-1:
        print('Not all GPU numbers selected are available, please restart the kernel, change your selection and execute again.')
        print('Available GPUs are:\n',availableGPUs)
        return list()
    else:
        if len(n)>1:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
            os.environ["CUDA_VISIBLE_DEVICES"]=str(n).replace('[','').replace(']','');
            from keras.utils import multi_gpu_model
            print("Multi GPU mode activated. Imported Keras multi GPU model module.")
            print(list(np.array(availableGPUs)[n]),'have been set.')
            return [j.split(": ")[1] for j in [i.split(" (")[0] for i in list(np.array(availableGPUs)[n])]]
        else:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
            os.environ["CUDA_VISIBLE_DEVICES"]=str(n[0]);
            print(availableGPUs[n[0]],'has been set.')
            return [availableGPUs[n[0]].split(" (")[0].split(": ")[1]]