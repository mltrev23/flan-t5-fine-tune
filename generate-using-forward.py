from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to('cuda')

inputs = 'Who is Steve Jobs?'

input_ids = tokenizer.encode(inputs, return_tensors='pt').to('cuda')
print(f"input_ids : {input_ids}")

#outputs = model.generate(input_ids, max_length=100)
outputs = model(input_ids, labels = input_ids).logits
print(f"outputs : {outputs}")
out = torch.argmax(outputs, dim=-1, keepdim=True)

print(tokenizer.decode(out, skip_special_tokens=True))
"""
import torch
        
input_text = "What is apple?"
# Encode the input text to tensor
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

outputs = input_ids
max_length = 100  # Define the maximum length of the sequence

decoder_start_token = tokenizer.pad_token_id
decoder_input_ids = torch.full(
    (input_ids.shape[0], 1), 
    decoder_start_token, 
    dtype=torch.long
).to('cuda')

with torch.no_grad():
    for _ in range(max_length):
        # Forward pass
        logits = model(outputs, decoder_input_ids=decoder_input_ids).logits

        # Get the logits of the last predicted token in the output
        next_token_logits = logits[:, -1, :]

        # Greedily select the next token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Check if the last token is EOS (end of sentence) token
        #if next_token.item() == tokenizer.eos_token_id:
        #    break

        # Append the predicted token to the output sequence
        #outputs = torch.cat([outputs, next_token], dim=1)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
print(decoder_input_ids[0])
generated_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
print(generated_text)
"""