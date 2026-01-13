import sys
import os
import pandas as pd
import numpy as np
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import random
import librosa
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_dataloader(dataset, batch_size=4, shuffle=True):
    import gc
    gc.collect() 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False, persistent_workers=False)

class AudioQADataset(Dataset):
    def __init__(self, data_dir, csv_data_path, split='train', like_test=False,
                 prefix_length=8, sample_data=None, model_type="google/gemma-2-2b",
                 target_audio_seconds=5, max_seq_length=256):
        super().__init__()

        sys.stdout.flush()
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ignore_index = self.tokenizer.pad_token_id
        self.sample_rate = 16000  
        self.target_audio_seconds = target_audio_seconds 

        data = pd.read_csv(csv_data_path, sep=',')
        self.audio_ids = data['ID'].values  
        self.questions = data['Question'].values  
        self.answers = data['Answer'].values  
        self.max_seqs_len = self.compute_max_seqs_len()

        self.train_setting = True if (split != 'test' and not like_test) else False
        self.prefix_len = prefix_length

    def __len__(self):
        return len(self.audio_ids)

    def compute_max_seqs_len(self):
        q_lens, a_lens = [], []
        for question in self.questions:
            q_lens.append(len(self.tokenizer.encode(question, add_special_tokens=False)))
        for answer in self.answers:
            a_lens.append(len(self.tokenizer.encode(str(answer), add_special_tokens=False)))
        return (int(np.mean(q_lens) + 2 * np.std(q_lens)), int(np.mean(a_lens) + 2 * np.std(a_lens)))

    def pad_sequences(self, index):
        m = [torch.tensor(self.tokenizer.encode('question: ', add_special_tokens=False)),
             torch.tensor(self.tokenizer.encode(' audio:', add_special_tokens=False)),
             torch.tensor(self.tokenizer.encode('answer ', add_special_tokens=False)),
             torch.tensor(self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False))]
        m_mask = [torch.ones(len(m[i])) for i in range(4)]

        q = torch.tensor(self.tokenizer.encode(self.questions[index], add_special_tokens=False))
        a = torch.tensor(self.tokenizer.encode(str(self.answers[index]), add_special_tokens=False))

        q, q_mask, leftover_tokens = self.make_padding(self.max_seqs_len[0], q, question=True)
        q_len = m[0].size(0) + q.size(0) + m[1].size(0)
        a, a_mask, _ = self.make_padding(self.max_seqs_len[1], a, leftover_tokens=leftover_tokens)

        if len((a == self.ignore_index).nonzero()) != 0:
            pad_start = (a == self.ignore_index).nonzero()[0]
        else:
            pad_start = []

        a = torch.cat((a, m[3])) if len(pad_start) == 0 else torch.cat((a[:pad_start], m[3], a[pad_start:]))
        q = torch.cat((m[0], q, m[1], torch.ones(self.prefix_len), m[2], a))

        q_mask = torch.cat((m_mask[0], q_mask, m_mask[1], torch.ones(self.prefix_len), m_mask[2], a_mask, m_mask[3]))
        return q, q_mask, q_len

    def make_padding(self, max_len, tokens, question=False, leftover_tokens=0):
        padding = max_len - tokens.size(0)
        if padding > 0:
            if question:
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0))
            else:
                padding_tokens = torch.tensor([self.ignore_index] * (padding + leftover_tokens))
                tokens = torch.cat((tokens, padding_tokens))
                mask = torch.zeros(max_len + leftover_tokens)
        elif padding == 0:
            mask = torch.ones(tokens.size(0)) if question else torch.zeros(tokens.size(0) + leftover_tokens)
            if not question:
                padding_tokens = torch.tensor([self.ignore_index] * leftover_tokens)
                tokens = torch.cat((tokens, padding_tokens))
        else:
            tokens = tokens[:max_len]
            mask = torch.ones(max_len) if question else torch.zeros(max_len + leftover_tokens)
            if not question:
                padding_tokens = torch.tensor([self.ignore_index] * leftover_tokens)
                tokens = torch.cat((tokens, padding_tokens))
        return tokens, mask, leftover_tokens

    def _read_audio_sample(self, audio_id):
        try:
            for file_name in os.listdir(self.data_dir):
                if file_name.startswith(str(audio_id)):
                    audio_path = os.path.join(self.data_dir, file_name)
                    waveform, _ = librosa.load(audio_path, sr=self.sample_rate)

                    frame_len = int(self.sample_rate / 10)
                    hop = int(frame_len / 2)  
                    waveform, _ = librosa.effects.trim(waveform, frame_length=frame_len, hop_length=hop)
                    return waveform

            raise FileNotFoundError(f"Audio file starting with ID {audio_id} not found.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None

    def _pre_process_audio_mel_t(self, audio, n_mels=64, f_min=50, f_max=8000, nfft=1024, hop=512):
        S = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
        S = librosa.power_to_db(S, ref=np.max)
        if S.max() != S.min():
            mel_db = (S - S.min()) / (S.max() - S.min())
        else:
            mel_db = S
            print("Warning in producing spectrogram!")
        return torch.tensor(mel_db.T, dtype=torch.float32)  

    def __getitem__(self, index):
        try:
            audio_id = self.audio_ids[index]
            waveform = self._read_audio_sample(audio_id)
            if waveform is None:
                raise ValueError(f"Unable to load audio for ID: {audio_id}")
            if self.train_setting:
                target_length = int(self.target_audio_seconds * self.sample_rate)
                waveform_length = waveform.shape[0]
                if waveform_length > target_length:
                    waveform = waveform[:target_length]
                elif waveform_length < target_length:
                    padding = target_length - waveform_length
                    waveform = np.pad(waveform, (0, padding), mode='constant') 
            spectrogram = self._pre_process_audio_mel_t(waveform)

            tokens, attention_mask, q_len = self.pad_sequences(index)

            if spectrogram is None or tokens is None or attention_mask is None or q_len is None:
                raise ValueError(f"Invalid data at index {index}")

            return spectrogram, tokens, attention_mask, q_len
        except Exception as e:
            print(f"Error at index {index}: {e}")
            return None
