from transformers import T5ForConditionalGeneration, T5Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'google/flan-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name, padding_side = 'left')
model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda')

# Calculate the total number of layers in the model
total_layers = len(model.decoder.block)
"""
for param in model.parameters():
    param.requires_grad = False
# Calculate the index of the layer to start unfreezing from (about one-third from the back)
start_index = total_layers - (total_layers // 3)
for layer in model.decoder.block[start_index:]:
    for param in layer.parameters():
        param.requires_grad = True
"""
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load unlabeled data and tokenize it
#unlabeled_data = open('bittensor.txt', 'r').readlines()
#tokenized_data = tokenizer(unlabeled_data, truncation=True, padding=True)

# Create training dataset for MLM
tokenizer.mask_token = tokenizer.eos_token
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='bittensor.txt', block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.5)
# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 300

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=dataset,
   tokenizer=tokenizer,
   data_collator=data_collator,
)

trainer.train()

input_text = "What is bittensor?"
# Encode the input text to tensor
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
# Generate text using the model. Adjust the max_length as needed.
output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
print(output_ids)

# Decode the generated ids to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Generated Text:", generated_text)
