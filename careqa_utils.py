# careqa_utils.py
import sys
import os
import torch
import numpy as np
import librosa
import logging
import contextlib
import io
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.models import AudioQAModel
from models.audio_encoder import initialize_pretrained_model
from models.prefix_mappers import TransformerMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CaReAQA")

class OperaWarningFilter(contextlib.redirect_stdout):
    def __init__(self):
        self._buffer = io.StringIO()
        super().__init__(self._buffer)

    def __exit__(self, exc_type, exc_val, exc_tb):
        output = self._buffer.getvalue()
        filtered_lines = [line for line in output.splitlines() 
                          if "No opera checkpoint provided" not in line]
        print("\n".join(filtered_lines), end="")
        return super().__exit__(exc_type, exc_val, exc_tb)

def load_careqa_model(repo_id, model_filename, llm_type, prefix_length=8):
    logger.info(f"Downloading model from {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    
    logger.info("Initializing model architecture...")
    with OperaWarningFilter():
        model = AudioQAModel(
            llm_type=llm_type,
            opera_checkpoint_path=None,
            prefix_length=prefix_length,
            clip_length=1,
            setting="lora",
            mapping_type="Transformer",
            fine_tune_opera=True,
            args=None
        ).eval().cuda()
    
    logger.info(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    return model, model_path

def preprocess_audio(audio_path, sr=16000):
    logger.info(f"Loading audio: {audio_path}")
    raw_audio, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=raw_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=64
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    audio_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0).cuda()
    return audio_tensor

def generate_answer(model, tokenizer, audio_tensor, question, prefix_length=8, audio_feature_dim=1280):
    logger.info("Extracting audio features...")
    with torch.no_grad():
        audio_features = model.audio_model.extract_feature(audio_tensor, dim=audio_feature_dim)
        projected_prefix = model.prefix_project(audio_features)

    logger.info("Preparing input for language model...")
    q_prefix = tokenizer.encode("question: ", add_special_tokens=False)
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    a_prefix = tokenizer.encode(" answer", add_special_tokens=False)

    input_tokens = q_prefix + q_tokens + a_prefix
    q_len = len(input_tokens)
    placeholder_ids = [tokenizer.eos_token_id] * prefix_length
    input_ids = torch.tensor([input_tokens + placeholder_ids], dtype=torch.long).to("cuda")
    attention_mask = torch.ones_like(input_ids)

    input_embeds = model.llm.get_input_embeddings()(input_ids)
    input_embeds[0, q_len : q_len + prefix_length] = projected_prefix[0]

    logger.info("Generating answer...")
    with torch.no_grad():
        output_ids = model.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if answer.lower().startswith("answer"):
        answer = answer[len("answer"):].strip()

    return answer

