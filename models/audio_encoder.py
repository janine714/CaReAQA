from transformers import ASTModel, ASTFeatureExtractor
import sys
sys.path.append('/home/twang/OPERA/src')
import os
import numpy as np
import torch
from model.models_mae import mae_vit_small  # OPERA's model import, no change needed
from model.models_cola import Cola  # OPERA's model import, no change needed
from huggingface_hub.file_download import hf_hub_download  # No change needed
from util import get_split_signal_librosa, get_entire_signal_librosa  # OPERA's util import, no change needed
import soundfile as sf
from transformers import ClapConfig, ClapModel

ENCODER_PATH_OPERA_CE_EFFICIENTNET = "cks/model/encoder-operaCE.ckpt"
ENCODER_PATH_OPERA_CT_HT_SAT = "cks/model/encoder-operaCT.ckpt"
ENCODER_PATH_OPERA_GT_VIT =  "cks/model/encoder-operaGT.ckpt"

def get_encoder_path(pretrain):
    encoder_paths = {
        "operaCT": ENCODER_PATH_OPERA_CT_HT_SAT,
        "operaCE": ENCODER_PATH_OPERA_CE_EFFICIENTNET,
        "operaGT": ENCODER_PATH_OPERA_GT_VIT
    }
    if not os.path.exists(encoder_paths[pretrain]):
        print("model ckpt not found, trying to download from huggingface")
        download_ckpt(pretrain)
    return encoder_paths[pretrain]

def download_ckpt(pretrain):
    model_repo = "evelyn0414/OPERA"
    model_name = "encoder-" + pretrain + ".ckpt"
    hf_hub_download(model_repo, model_name, local_dir="cks/model")


def initialize_pretrained_model(pretrain, device="cpu"):
    if pretrain == "ast":
        # Load AST model
        model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", trust_remote_code=True).to(device)
        # Load AST feature extractor
        feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        return model, feature_extractor
    elif pretrain == "operaCT":
        model = Cola(encoder="htsat")
    elif pretrain == "operaCE":
        model = Cola(encoder="efficientnet")
    elif pretrain == "operaGT":
        model = mae_vit_small(
            norm_pix_loss=False,
            in_chans=1,
            audio_exp=True,
            img_size=(256, 64),
            alpha=0.0,
            mode=0,
            use_custom_patch=False,
            split_pos=False,
            pos_trainable=False,
            use_nce=False,
            decoder_mode=1,
            mask_2d=False,
            mask_t_prob=0.7,
            mask_f_prob=0.3,
            no_shift=False
        ).float()
    elif pretrain == "clap":
        # Load CLAP model
        model = ClapModel.from_pretrained("laion/clap-htsat-fused", trust_remote_code=True).to(device)
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        return model, processor
    else:
        raise NotImplementedError(f"Model not exist: {pretrain}, please check the parameter.")

    return model
