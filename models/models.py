from typing import Tuple, Optional, Union
import sys
import os
import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from torch.nn import functional as nnf

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, VeraConfig, get_peft_model, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from .prefix_mappers import MLP, TransformerMapper, LinearMapper
from .audio_encoder import initialize_pretrained_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioQAModel(nn.Module):
    def __init__(self,
                 llm_type,
                 opera_checkpoint_path,
                 prefix_length=8,
                 clip_length=8,
                 prefix_size=1280,
                 num_layers=8,
                 setting="lora",
                 mapping_type="MLP",
                 fine_tune_opera=True,
                 args=None):
        super(AudioQAModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_type)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize the audio model
        self.audio_model = initialize_pretrained_model(pretrain='operaCE')

        # Load checkpoint only if a valid path is provided
        if opera_checkpoint_path:
            print(f"Loading opera checkpoint from {opera_checkpoint_path}...")
            checkpoint = torch.load(opera_checkpoint_path, map_location='cuda')
            self.audio_model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("Opera checkpoint loaded successfully.")
        else:
            print("No opera checkpoint provided. Using default weights.")

        # Handle fine-tuning settings
        if not fine_tune_opera:
            self.audio_model = self.audio_model.encoder
            for param in self.audio_model.parameters():
                param.requires_grad = False
            self.audio_model.eval()

        # Initialize the LLM
        self.llm_type = llm_type
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_type,
                                                        device_map="auto",
                                                        torch_dtype="auto",
                                                        trust_remote_code=True)
        self.setting = setting

        # Parameter-efficient fine-tuning configurations
        if setting == "vera":
            peft_config = VeraConfig(
                task_type="CAUSAL_LM",
                r=args.vera_r if args else 256,
                projection_prng_key=args.vera_projection_prng_key if args else 42,
                save_projection=True,
                vera_dropout=args.vera_dropout if args else 0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.llm = get_peft_model(self.llm, peft_config)
        elif setting == "lora":
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.llm = get_peft_model(self.llm, peft_config)
        elif setting == "prefixtuning":
            peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=30)
            self.llm = get_peft_model(self.llm, peft_config)
        elif setting == "p_tuning":
            peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=30)
            self.llm = get_peft_model(self.llm, peft_config)
        elif setting == "prompttuning":
            peft_config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=30)
            self.llm = get_peft_model(self.llm, peft_config)
        elif setting == 'frozen':
            for param in self.llm.transformer.parameters():
                param.requires_grad = False
        elif setting == "mapping_only":
            for param in self.audio_model.parameters():
                param.requires_grad = False
            for param in self.llm.parameters():
                param.requires_grad = False

        if setting in ["lora", "vera"]:
            self.llm_embedding_size = self.llm.base_model.model.model.embed_tokens.weight.shape[1]
        else:
            self.llm_embedding_size = self.llm.model.embed_tokens.weight.shape[1]

        # Create prefix projection layer based on mapping type
        if mapping_type == "MLP":
            self.prefix_project = MLP((
                prefix_size,
                (self.llm_embedding_size * prefix_length) // 2,
                self.llm_embedding_size * prefix_length,
                self.llm_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.prefix_project = TransformerMapper(
                prefix_size,
                self.llm_embedding_size,
                prefix_length,
                clip_length, num_layers)
        elif mapping_type == "Linear":
            self.prefix_project = LinearMapper(
                input_size=prefix_size,
                output_size=self.llm_embedding_size * prefix_length)
        else:
            raise ValueError("Select a valid mapping type: MLP, Transformer, or Linear")

        self.prefix_length = prefix_length

    def forward(self, spectrogram, input_ids, attention_mask, q_len):
        try:
            batch_size = spectrogram.size(0)

            prefix = self.audio_model(spectrogram)
            prefix_projections = self.prefix_project(prefix).view(
                -1, self.prefix_length, self.llm_embedding_size)

            if self.setting in ["lora", "vera"]:
                embedding = self.llm.base_model.model.model.embed_tokens(input_ids)
            else:
                embedding = self.llm.model.embed_tokens(input_ids)

            for b in range(batch_size):
                q_length = q_len[b].item()
                if q_length + self.prefix_length <= embedding.shape[1]:
                    embedding[b, q_length:q_length + self.prefix_length, :] = prefix_projections[b]
                else:
                    raise ValueError(f"Cannot assign prefix projection for batch {b}, q_len={q_length}, prefix_length={self.prefix_length}, embedding length={embedding.shape[1]}")

            output = self.llm(inputs_embeds=embedding, attention_mask=attention_mask)
            return output

        except ValueError as ve:
            logger.error(f"ValueError in forward pass: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    def generate(self, prefix, tokens, mask, q_len):
        with torch.no_grad():
            try:
                prefix = torch.unsqueeze(prefix, 0)
                prefix = self.audio_model(prefix)

                prefix_projections = self.prefix_project(prefix).view(
                    self.prefix_length, self.llm_embedding_size)

                if self.setting in ["lora", "vera"]:
                    embedding_txt = self.llm.base_model.model.model.embed_tokens(tokens)
                else:
                    embedding_txt = self.llm.model.embed_tokens(tokens)

                embedding_txt[q_len:q_len + self.prefix_length, :] = prefix_projections
                outputs = self.llm(inputs_embeds=embedding_txt, attention_mask=mask)

                return embedding_txt

            except ValueError as ve:
                logger.error(f"ValueError in generate method: {ve}")
                raise
            except Exception as e:
                logger.error(f"Error in generate method: {e}")
                raise
