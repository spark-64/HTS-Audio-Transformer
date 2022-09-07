# import basic packages
import os
import numpy as np
import librosa

# Load the model package
import torch
import esc_config as config
from model.htsat import HTSAT_Swin_Transformer


# in the notebook, we only can use one GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# infer the single data to check the result
# get a model you saved
model_path = 'HTSAT_ESC_exp=1_fold=3_acc=0.973.ckpt'

class Audio_Classification:
    def __init__(self, model_path, config):
        super().__init__()

        self.device = torch.device('cuda')
        self.sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config = config,
            depths = config.htsat_depth,
            embed_dim = config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )
        ckpt = torch.load(model_path, map_location="cpu")
        temp_ckpt = {}
        for key in ckpt["state_dict"]:
            temp_ckpt[key[10:]] = ckpt['state_dict'][key]
        self.sed_model.load_state_dict(temp_ckpt)
        self.sed_model.to(self.device)
        self.sed_model.eval()


    def predict(self, audiofile):

        if audiofile:
            waveform, sr = librosa.load(audiofile, sr=32000)

            with torch.no_grad():
                x = torch.from_numpy(waveform).float().to(self.device)
                output_dict = self.sed_model(x[None, :], None, True)
                pred = output_dict['clipwise_output']
                pred_post = pred[0].detach().cpu().numpy()
                pred_label = np.argmax(pred_post)
                pred_prob = np.max(pred_post)
            return pred_label, pred_prob

# Inference
Audiocls = Audio_Classification(model_path, config)

# pick any audio you like in the ESC-50 testing set (cross-validation)
pred_label, pred_prob = Audiocls.predict("1-137-A-32.wav")

print('Audiocls predict output: ', pred_label, pred_prob)