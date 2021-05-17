import streamlit as st
import librosa
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from architecture import Network
import numpy as np

### Loading model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model

network_best = load_checkpoint('checkpoint.pth')
network_best= network_best



html_temp = """
    <div style="background-color:purple;padding:10px">
    <h1 style="color:white;text-align:center;">Tool Sound Classifier </h1>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

### GEtting Input
file = st.sidebar.file_uploader("Please Upload Mp3 Audio File Here or Use Demo Of App Below using Preloaded Music",type=["wav","mp3"])

# def save_uploadedfile(uploadedfile):
#      with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
#          f.write(uploadedfile.getbuffer())
#          st.success("Saved File:{} to tempDir".format(uploadedfile.name))
#          return("Saved File:{} to tempDir".format(uploadedfile.name))

### Input Preprocessing
def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    # file_path= librosa.ex(str(file_path.name)) 
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    Tx= ToTensor()(spec_scaled)
    out = F.interpolate(Tx, size=128)  #The resize operation on tensor.
    return out


### Getting Output
def output(file):
    xdb= get_melspectrogram_db(file)
    out= np.array(spec_to_image(xdb), dtype = 'float32')*15
    out= out.reshape(1, 128, 128)
    out= np.array([out.tolist()], dtype = 'float32')
    out= torch.from_numpy(out).float()

    out_dict= {0: "Chainsaw", 1:"Drill", 2:"Hammer", 3:"Horn", 4:"Sword"}
    for i in range(len(network_best(out)[0])):
        k=network_best(out).max()
        if k == network_best(out)[0][i]:
            return out_dict[i]

html_temp1 = """
    <div>
    <h2 style="color:black;text-align:left;">Prediction: </h2>
    </div>
    """

if file is not None:
    # # st.text(output(str(file.name)))
    # file_details = {"FileName":file.name,"FileType":file.type}
    # # save_uploadedfile(file)

    # st.text(output(save_uploadedfile(file)))
    st.markdown(html_temp1,unsafe_allow_html=True)
    st.text(output(file))