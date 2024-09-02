import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summary_text(origin_text: str) -> str:
    text = origin_text.replace('\n', '')
    raw_input_ids = tokenizer.encode(origin_text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
    input_ids = torch.tensor([input_ids]).to(device)
    
    summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
    summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    
    return summary