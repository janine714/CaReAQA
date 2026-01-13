from careqa_utils import load_careqa_model, preprocess_audio, generate_answer
from transformers import AutoTokenizer
import logging

logger = logging.getLogger("CaReAQA")

def main():
    repo_id = "tsnngw/CaReAQA"
    model_filename = "model.pt"
    audio_path = "/home/twang/cross_modal_alignment/datasets/KAUH/AudioFiles/BP54_heart failure,Crep,P R L ,73,F.wav"
    question = "Where were the abnormal sounds detected?"
    #audio_path = "/home/twang/circor/69155_PV.wav"
    #question = "Where is the murmur most audible?"
    
    #audio_path = "/home/twang/circor/2530_AV.wav"
    #question = "Is there a murmur present in the cardiac auscultation findings?"
    

    prefix_length = 8
    audio_feature_dim = 1280
    base_llama_model = "meta-llama/Llama-3.2-3B"

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_llama_model, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, _ = load_careqa_model(repo_id, model_filename, base_llama_model, prefix_length)
    audio_tensor = preprocess_audio(audio_path)

    answer = generate_answer(model, tokenizer, audio_tensor, question, prefix_length, audio_feature_dim)

    logger.info("=" * 50)
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")
    logger.info("=" * 50)

    return answer

if __name__ == "__main__":
    main()
