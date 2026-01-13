import os
import torch
from tqdm import tqdm
import re
import csv
from transformers import AutoTokenizer
from evaluate import load
from utils import generate_beam
from dataloader import AudioQADataset
from torch.utils.data import DataLoader

# Load evaluation metrics
bert_score = load("bertscore")
meteor = load("meteor")

def clean_generated_answer(text):
    if "answer" in text:
        answer_start_idx = text.find("answer") + len("answer")
        text = text[answer_start_idx:].strip()
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Collapse multiple spaces into one
    text = ' '.join(text.split())
    # Remove everything after a period if it seems like extra trailing text
    text = re.sub(r'\.\s.*', '.', text)
    # Remove repeated parentheses and other stray characters
    text = re.sub(r'[\(\)]+', '', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    return text

def eval_llm_open_ended(model, dataloader, args):
    model.eval()
    model = model.cuda().half()  # Ensure model uses consistent float16 precision
    tokenizer = AutoTokenizer.from_pretrained(args.llm_type)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    true_answers, generated_answers = [], []
    shift = 10 if args.setting in ["p_tuning", "prompttuning"] else 0
    original_dataset = dataloader.dataset
    subset_indices = original_dataset.indices

    results = []  # List to store results for CSV output

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            spectrograms, input_ids, attention_masks, q_lens = batch
            
            # Ensure q_lens is an integer tensor
            if isinstance(q_lens, int):
                q_lens = torch.tensor([q_lens] * input_ids.size(0), dtype=torch.long)
            else:
                q_lens = torch.tensor(q_lens, dtype=torch.long)
            
            # Move tensors to GPU and set precision
            spectrograms = spectrograms.half().cuda()
            input_ids = input_ids.cuda().long()
            attention_masks = attention_masks.half().cuda()
            q_lens = q_lens.cuda()

            with torch.amp.autocast("cuda"):
                outputs = model(spectrograms, input_ids, attention_masks, q_lens)
                logits = outputs.logits

            condensed_logits = logits[:, shift + model.prefix_length:-1]
            predicted_ids = torch.argmax(condensed_logits, dim=-1)

            for i in range(predicted_ids.size(0)):
                out_text = tokenizer.decode(predicted_ids[i], skip_special_tokens=True, errors='ignore')
                candidate = clean_generated_answer(out_text)
                ref_idx = batch_idx * dataloader.batch_size + i
                original_idx = subset_indices[ref_idx]
                reference = original_dataset.dataset.answers[original_idx]

                # Retrieve audio ID and question from the dataset
                audio_id = original_dataset.dataset.audio_ids[original_idx]
                question = original_dataset.dataset.questions[original_idx]

                if len(candidate) == 0:
                    candidate = "No valid answer generated."

                print(f"\nSample {ref_idx + 1}:")
                print(f"Audio ID: {audio_id}")
                print(f"Question: {question}")
                print(f"Reference (Ground Truth): {reference}")
                print(f"Generated Answer (Cleaned): {candidate}\n")

                true_answers.append(reference)
                generated_answers.append(candidate)

                # Append the result to the list
                results.append({
                    'audio_id': audio_id,
                    'question': question,
                    'ground_truth': reference,
                    'generated_answer': candidate
                })

    if len(true_answers) == 0 or len(generated_answers) == 0:
        print("No valid samples to evaluate.")
        return 0.0, 0.0

    try:
        bert_scores = bert_score.compute(predictions=generated_answers, references=true_answers, model_type='bert-base-uncased')
        bert_avg = sum(bert_scores['f1']) / len(bert_scores['f1'])
    except Exception as e:
        print(f"BERTScore computation failed: {e}")
        bert_avg = 0.0

    try:
        meteor_score = meteor.compute(predictions=generated_answers, references=true_answers)['meteor']
    except Exception as e:
        print(f"METEOR computation failed: {e}")
        meteor_score = 0.0

    print("\nFinal Evaluation Metrics:")
    print(f"Average BERTScore F1: {bert_avg}")
    print(f"METEOR Score: {meteor_score}")

    # Write the results to a CSV file
    with open('results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['audio_id', 'question', 'ground_truth', 'generated_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return bert_avg, meteor_score
