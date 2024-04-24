from torch.autograd import Variable
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import TextDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch.nn.functional as F

class T5FineTuner(nn.Module):
    def __init__(self):
        super(T5FineTuner, self).__init__()
        self.t5model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to('cuda')
        for param in self.t5model.parameters():
            param.requires_grad = False
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32128, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 32128),
            nn.ReLU(),
            nn.Linear(32128, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 32128),
            nn.ReLU(),
            nn.Linear(32128, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 32128),
            nn.ReLU(),
            nn.Linear(32128, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 32128),
        )
        for param in self.linear_relu_stack.parameters():
            param.requires_grad = True
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        outputs = self.t5model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        res = outputs  # Get the logits directly from the T5 model
        #res.logits = self.linear_relu_stack(outputs.logits)  # Apply linear layers to logits
        res.logits = Variable(outputs.logits, requires_grad = True)
        return res

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5FineTuner().to('cuda')

class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path) as f:
            self.texts = f.readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        cur = self.texts[idx]
        inputs = self.tokenizer.encode_plus(cur, add_special_tokens=True, padding='max_length', max_length=self.max_length, return_tensors='pt', return_attention_mask=True, return_token_type_ids=False).to('cuda')
        return inputs


train_path = 'bittensor.txt'
train_dataset = MyDataset(tokenizer=tokenizer, file_path=train_path)
num_epochs = 5
data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#print(train_dataset.examples)
from torch.optim import Adam

# Create an optimizer with only the new parameters
optimizer = Adam(model.linear_relu_stack.parameters(), lr=1e-4)
print(model.parameters())
print('---------------------')

# Example pseudo-training loop
for epoch in range(num_epochs):
    print(f'-------------------------{epoch}-------------------------')
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['input_ids']
        originshape = input_ids.shape
        input_ids = input_ids.reshape(originshape[0], originshape[2])
        attention_mask = attention_mask.reshape(originshape[0], originshape[2])
        labels = labels.reshape(originshape[0], originshape[2])
        optimizer.zero_grad()
        print(f'inputs shape : {input_ids.shape}')
        print(f'inputs : {input_ids}')
        print(f'labels shape : {labels.shape}')
        outputs = model(input_ids, attention_mask = attention_mask, labels=labels)
        
        loss = outputs.loss
        
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

input_text = "What is apple?"
# Encode the input text to tensor
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
# Generate text using the model. Adjust the max_length as needed.
#output_ids = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
with torch.no_grad():
    outputs = model(input_ids, labels = input_ids)
#print(outputs.logits.shape)
softmax = nn.Softmax(dim=2)
probabilities = softmax(outputs.logits)
#print(probabilities.shape)
output_ids = torch.argmax(probabilities, dim = 2)

# Decode the generated ids to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Input Text:", input_text)
print("Generated Text:", generated_text)
"""
outputs = input_ids
max_length = 50  # Define the maximum length of the sequence
with torch.no_grad():
    for _ in range(max_length):
        # Forward pass
        logits = model(outputs).logits

        # Get the logits of the last predicted token in the output
        next_token_logits = logits[:, -1, :]

        # Greedily select the next token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Check if the last token is EOS (end of sentence) token
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append the predicted token to the output sequence
        outputs = torch.cat([outputs, next_token], dim=1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
