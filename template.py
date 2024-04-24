from transformers import T5ForConditionalGeneration, T5Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'google/flan-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name, padding_side = 'right')
model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda')
"""
# Calculate the total number of layers in the model
total_layers = len(model.decoder.block)

# Calculate the index of the layer to start unfreezing from (about one-third from the back)
start_index = total_layers - (total_layers // 3)
for layer in model.decoder.block[start_index:]:
    for param in layer.parameters():
        param.requires_grad = True
"""
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load unlabeled data and tokenize it
#unlabeled_data = open('bittensor.txt', 'r').readlines()
#tokenized_data = tokenizer(unlabeled_data, truncation=True, padding=True)

# Create training dataset for MLM
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='bittensor.txt', block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.15)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()


input_text = "What is bittensor?"
# Encode the input text to tensor
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
# Generate text using the model. Adjust the max_length as needed.
output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

# Decode the generated ids to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Generated Text:", generated_text)
