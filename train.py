from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import TrainingArguments, Trainer

class Poem_Dataset(Dataset):
    def __init__(self, train_dataset, tokenizer):
        self.dataset = train_dataset
        self.tokenizer = tokenizer
        self.tokenized_dataset = []
        
        for data in self.dataset:
            data = "<s>" + data + "</s>"
            tokenized_data = self.tokenizer(data, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            self.tokenized_dataset.append(tokenized_data)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.tokenized_dataset[index]
        input_ids = item["input_ids"].squeeze(0)
        label_ids = item["input_ids"].squeeze(0)
        attention_mask = item["attention_mask"].squeeze(0)
        return {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" : label_ids
        }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
        bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    
    file_path = "poem.csv"
    
    dataset = pd.read_csv(file_path)
    train_dataset = list(dataset["시"])
    train_data = Poem_Dataset(train_dataset, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="output/",
        overwrite_output_dir=True,
        logging_steps=100,
        save_steps=100,
        save_total_limit=1,
        learning_rate=1e-05,
        per_device_train_batch_size=8,
        num_train_epochs=20,
        lr_scheduler_type="linear",
        warmup_steps=100,
        seed=42
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data
    )
    
    trainer.train()