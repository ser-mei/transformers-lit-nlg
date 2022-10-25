from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from datasets import load_dataset

#Split de dataset para entrenamiento y testeo
data_files ={"train": "borges-sin-sl.txt", "test": "Elhombreenelumbral.txt"}

#Carga el dataset desde archivo de texto
borgesDataset = load_dataset("text", data_files=data_files)

#print(type(borgesDataset["train"][0]))
#print(borgesDataset)
#print(borgesDataset["train"][:2]["text"])


context_length = 128

#Se carga el tokenizador del modelo "gpt2-Spanish"
tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")


#outputs = tokenizer(
#    borgesDataset["train"][:-1]["text"],
#    truncation = True,
#    max_length = context_length,
#    return_overflowing_tokens = True,
#    return_length = True,
#)

#print(f"Input IDs length: {len(outputs['input_ids'])}")
#print(f"Input chunk lengths: {(outputs['length'])}")
#print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")


#Función para tokenizar y organizar los batchs
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


#Dataset tokenizado
tokenized_datasets = borgesDataset.map(
    tokenize, batched=True, remove_columns=borgesDataset["train"].column_names
)
print(tokenized_datasets)

#Condiguración del modelo
config = AutoConfig.from_pretrained(
    'DeepESP/gpt2-spanish',
    vocab_size = len(tokenizer),
    n_ctx = context_length,
    bos_token = tokenizer.bos_token,
    eos_token = tokenizer.eos_token,
)

model = AutoModelForCausalLM.from_config(config)

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)

out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")
    

args = TrainingArguments(
    output_dir="borges-ds",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=1_000,
    logging_steps=1_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=1_000,
    fp16=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()