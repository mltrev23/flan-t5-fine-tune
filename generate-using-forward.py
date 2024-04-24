from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

inputs = 'What is apple?'

input_ids = tokenizer.encode(inputs, return_tensors='pt')

outputs = model.generate(input_ids, max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))