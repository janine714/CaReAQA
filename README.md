# CaReAQA

CaReAQA (Cardiac & Respiratory Audio Question Answering) is an audio-language model for **open-ended diagnostic reasoning** over medical auscultation audio (heart/lung sounds) conditioned on natural-language questions.

- Paper (CHIL 2025 / PMLR): https://proceedings.mlr.press/v287/wang25b.html  
- arXiv: https://arxiv.org/abs/2505.01199  
- Pretrained CaReAQA weights (Hugging Face): https://huggingface.co/tsnngw/CaReAQA  

> ⚠️ Medical disclaimer: This project is for research and educational use only. It is **not** medical advice and must **not** be used for clinical decisions.

---

## What you need

This repo depends on three things:

1. **This GitHub repo** 
2. **CaReAQA checkpoint** on Hugging Face (`tsnngw/CaReAQA`, file `CaReAQAmodel.pt`)
3. **OPERA** (audio encoder code + checkpoint), from the official OPERA repo

In addition, the default base LLM is `meta-llama/Llama-3.2-3B`, which is **license-gated** on Hugging Face.

---

## Installation

### 1) Clone this repo

```bash
git clone https://github.com/janine714/CaReAQA.git
cd CaReAQA
```

### 2) Create an environment and install dependencies


---

## Hugging Face access (required)

This project loads:
- CaReAQA checkpoint from Hugging Face, and
- the base LLaMA tokenizer/model from `meta-llama/*` (requires accepting Meta’s license on Hugging Face)

---

## OPERA dependency (required)

OPERA repo:
https://github.com/evelyn0414/OPERA

### 1) Clone OPERA

```bash
git clone https://github.com/evelyn0414/OPERA.git
```

### 2) Make OPERA importable

Set OPERA_ROOT to your cloned OPERA directory

### 3) Download OPERA checkpoint(s)

Follow OPERA’s official instructions to download the required pretrained checkpoint(s). The exact filenames depend on the OPERA release you use.

---

## Pretrained CaReAQA checkpoint

The CaReAQA weights are on Hugging Face:

- Repo: `tsnngw/CaReAQA`
- File: `CaReAQAmodel.pt`

Your code should download it via `huggingface_hub.hf_hub_download()`.

---

## Quickstart: run inference

Main runnable script:

```bash
python scripts/inference.py
```

By default, `scripts/inference.py` may contain a hard-coded example `audio_path` and `question`.
Edit those variables inside `scripts/inference.py` and rerun.

---

## Minimal Python usage 

```python
from transformers import AutoTokenizer
from careqa_utils import load_careqa_model, preprocess_audio, generate_answer

repo_id = "tsnngw/CaReAQA"
model_filename = "CaReAQAmodel.pt"

audio_path = "/path/to/audio.wav"
question = "Where is the murmur most audible?"

prefix_length = 8
audio_feature_dim = 1280
base_llama_model = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(base_llama_model, token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model, _ = load_careqa_model(repo_id, model_filename, base_llama_model, prefix_length)
audio_tensor = preprocess_audio(audio_path)

answer = generate_answer(model, tokenizer, audio_tensor, question, prefix_length, audio_feature_dim)
print("Q:", question)
print("A:", answer)
```

---

## Citation

```bibtex
@misc{wang2025careaqacardiacrespiratoryaudio,
      title={CaReAQA: A Cardiac and Respiratory Audio Question Answering Model for Open-Ended Diagnostic Reasoning}, 
      author={Tsai-Ning Wang and Lin-Lin Chen and Neil Zeghidour and Aaqib Saeed},
      year={2025},
      eprint={2505.01199},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.01199}, 
}
```

---

## Acknowledgements

- OPERA audio encoder: https://github.com/evelyn0414/OPERA
- Meta LLaMA models on Hugging Face: https://huggingface.co/meta-llama
