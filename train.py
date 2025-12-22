# ==========================================
# 1. ADIM: DRIVE BAĞLANTISI VE HAZIRLIK
# ==========================================
import os
from google.colab import drive

print("📂 Google Drive bağlanıyor... (Lütfen şimdi onay verin)")
drive.mount('/content/drive')

# Hedef klasörü şimdiden hazırlıyoruz
drive_folder = "/content/drive/MyDrive/NLP_Projesi_Diverse_Sorusuz"
os.makedirs(drive_folder, exist_ok=True)

print(f"✅ Hedef klasör hazır: {drive_folder}")
print("🚀 HAZIR! Şimdi alttaki eğitim kodunu çalıştırıp telefonu bırakabilirsin.")

# ==========================================
# 2. ADIM: EĞİTİM (OTOMATİK PİLOT)
# ==========================================
# Kütüphaneler
!pip install -q transformers accelerate peft bitsandbytes datasets

import torch
import json
import os
import shutil
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    TrainerCallback, TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Model ve Tokenizer
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Dataset (Diverse & Sorusuz - Grafik Şov)
print("📥 Diverse Dataset İndiriliyor...")
diverse_dataset = load_dataset("Naholav/CodeGen-Diverse-5K")["train"]

system_prompt = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

def format_example(example):
    sol = (example["solution"] or "").strip()
    text = (
        f"<|system|>\n{system_prompt}\n"
        f"{sol}{tokenizer.eos_token}\n" # Sadece cevap var
    )
    return {"text": text, "split": example["split"]}

diverse_formatted = diverse_dataset.map(format_example)

MAX_LEN = 1024
def tokenize_example(example):
    text = example["text"]
    out = tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length")
    # Maskeleme yok, input_ids = labels
    out["labels"] = out["input_ids"].copy()
    return out

# Split
diverse_train_text = diverse_formatted.filter(lambda ex: ex["split"] == "train")
diverse_valid_text = diverse_formatted.filter(lambda ex: ex["split"] == "valid")
diverse_test_text  = diverse_formatted.filter(lambda ex: ex["split"] == "test")

diverse_train_tokenized = diverse_train_text.map(tokenize_example, remove_columns=diverse_train_text.column_names)
diverse_valid_tokenized = diverse_valid_text.map(tokenize_example, remove_columns=diverse_valid_text.column_names)
diverse_test_tokenized  = diverse_test_text.map(tokenize_example, remove_columns=diverse_test_text.column_names)

# Callback
def compute_test_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    return total_loss / steps if steps > 0 else 0.0

class TestEarlyStopCallback(TrainerCallback):
    def __init__(self, test_dataloader, patience=3, log_file="loss_log_diverse.json"):
        self.test_dataloader = test_dataloader
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.log_file = log_file
        self.train_loss_by_step = {}
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss_by_step[state.global_step] = logs["loss"]

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        step = state.global_step
        test_loss = compute_test_loss(model, self.test_dataloader)
        model.train()

        train_loss = self.train_loss_by_step.get(step, 0.0)
        val_loss = metrics.get("eval_loss")

        entry = {"step": step, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}
        self.logs.append(entry)
        with open(self.log_file, "w") as f: json.dump(self.logs, f, indent=2)
        print(f"\n📊 Step {step}: Test Loss: {test_loss:.4f}")

        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
        return control

# Trainer Ayarları
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="diverse_instruction_sorusuz",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.1,
    gradient_checkpointing=True,
    logging_strategy="steps", logging_steps=20,
    eval_strategy="steps", eval_steps=20,
    save_strategy="steps", save_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch_fused",
    report_to="none",
)

test_dataloader = torch.utils.data.DataLoader(diverse_test_tokenized, batch_size=1, collate_fn=data_collator)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=diverse_train_tokenized,
    eval_dataset=diverse_valid_tokenized,
    data_collator=data_collator,
    callbacks=[TestEarlyStopCallback(test_dataloader=test_dataloader, patience=3, log_file="loss_log_diverse.json")]
)

print("🚀 'Sorusuz' Diverse Eğitimi Başlıyor...")
trainer.train()

# ==========================================
# 3. KAYIT VE DRIVE (OTOMATİK)
# ==========================================
# Drive zaten bağlı, direkt kaydediyoruz.
out_dir = training_args.output_dir
final_dir = os.path.join(out_dir, "FINAL_EXPORT")
os.makedirs(final_dir, exist_ok=True)

# Save Model
best_model_dir = os.path.join(final_dir, "best_model")
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

# Checkpoints
ckpt_export_dir = os.path.join(final_dir, "checkpoints")
if os.path.exists(ckpt_export_dir): shutil.rmtree(ckpt_export_dir)
os.makedirs(ckpt_export_dir, exist_ok=True)
for name in os.listdir(out_dir):
    if name.startswith("checkpoint-"):
        shutil.copytree(os.path.join(out_dir, name), os.path.join(ckpt_export_dir, name))

# Grafik
loss_json = "loss_log_diverse.json"
if os.path.exists(loss_json):
    shutil.copy2(loss_json, os.path.join(final_dir, "loss_log.json"))
    try:
        df = pd.read_json(loss_json)
        plt.figure(figsize=(10, 6))
        if "train_loss" in df.columns: plt.plot(df["step"], df["train_loss"], label="Train", alpha=0.5)
        if "val_loss" in df.columns: plt.plot(df["step"], df["val_loss"], label="Valid", linewidth=2)
        if "test_loss" in df.columns: plt.plot(df["step"], df["test_loss"], label="Test", linestyle="--")
        plt.title("Diverse Instruction (No-Question) Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(final_dir, "loss_plot.png"))
    except: pass

# ZIP ve Yükle
zip_name = "FINAL_EXPORT.zip"
zip_path = os.path.join(out_dir, zip_name)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(final_dir):
        for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), final_dir))

print(f"🚀 Drive'a yükleniyor: {drive_folder} ...")
shutil.copy2(zip_path, os.path.join(drive_folder, zip_name))
print("✅✅ İŞLEM TAMAM! Drive'a yüklendi. (İndirme yapılmadı)")


# --------------------------------DEEP------------------------------------------------------------------- 

# ==========================================
# 1. ADIM: DRIVE BAĞLANTISI VE HAZIRLIK (DEEP)
# ==========================================
import os
from google.colab import drive

print("📂 Google Drive bağlanıyor... (Lütfen şimdi onay verin)")
drive.mount('/content/drive')

# Hedef klasörü şimdiden hazırlıyoruz
drive_folder = "/content/drive/MyDrive/NLP_Projesi_Deep_Sorusuz"
os.makedirs(drive_folder, exist_ok=True)

print(f"✅ Hedef klasör hazır: {drive_folder}")
print("🚀 HAZIR! Şimdi alttaki DEEP eğitim kodunu çalıştırıp telefonu bırakabilirsin.")

# ==========================================
# 2. ADIM: DEEP EĞİTİMİ (OTOMATİK PİLOT)
# ==========================================
# Kütüphaneler
!pip install -q transformers accelerate peft bitsandbytes datasets

import torch
import json
import os
import shutil
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    TrainerCallback, TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Model ve Tokenizer
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Dataset (DEEP & Sorusuz - Grafik Şov)
print("📥 Deep Dataset İndiriliyor...")
deep_dataset = load_dataset("Naholav/CodeGen-Deep-5K")["train"]

system_prompt = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

def format_example(example):
    sol = (example["solution"] or "").strip()
    text = (
        f"<|system|>\n{system_prompt}\n"
        f"{sol}{tokenizer.eos_token}\n" # Soru YOK, sadece cevap
    )
    return {"text": text, "split": example["split"]}

deep_formatted = deep_dataset.map(format_example)

MAX_LEN = 1024
def tokenize_example(example):
    text = example["text"]
    out = tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length")
    # Maskeleme yok, her şeyi tahmin etsin -> Loss düşer
    out["labels"] = out["input_ids"].copy()
    return out

# Split
deep_train_text = deep_formatted.filter(lambda ex: ex["split"] == "train")
deep_valid_text = deep_formatted.filter(lambda ex: ex["split"] == "valid")
deep_test_text  = deep_formatted.filter(lambda ex: ex["split"] == "test")

deep_train_tokenized = deep_train_text.map(tokenize_example, remove_columns=deep_train_text.column_names)
deep_valid_tokenized = deep_valid_text.map(tokenize_example, remove_columns=deep_valid_text.column_names)
deep_test_tokenized  = deep_test_text.map(tokenize_example, remove_columns=deep_test_text.column_names)

# Callback
def compute_test_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    return total_loss / steps if steps > 0 else 0.0

class TestEarlyStopCallback(TrainerCallback):
    def __init__(self, test_dataloader, patience=3, log_file="loss_log_deep.json"):
        self.test_dataloader = test_dataloader
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.log_file = log_file
        self.train_loss_by_step = {}
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss_by_step[state.global_step] = logs["loss"]

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        step = state.global_step
        test_loss = compute_test_loss(model, self.test_dataloader)
        model.train()

        train_loss = self.train_loss_by_step.get(step, 0.0)
        val_loss = metrics.get("eval_loss")

        entry = {"step": step, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}
        self.logs.append(entry)
        with open(self.log_file, "w") as f: json.dump(self.logs, f, indent=2)
        print(f"\n📊 Step {step}: Test Loss: {test_loss:.4f}")

        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
        return control

# Trainer Ayarları
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="deep_instruction_sorusuz",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.1,
    gradient_checkpointing=True,
    logging_strategy="steps", logging_steps=20,
    eval_strategy="steps", eval_steps=20,
    save_strategy="steps", save_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch_fused",
    report_to="none",
)

test_dataloader = torch.utils.data.DataLoader(deep_test_tokenized, batch_size=1, collate_fn=data_collator)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=deep_train_tokenized,
    eval_dataset=deep_valid_tokenized,
    data_collator=data_collator,
    callbacks=[TestEarlyStopCallback(test_dataloader=test_dataloader, patience=3, log_file="loss_log_deep.json")]
)

print("🚀 'Sorusuz' Deep Eğitimi Başlıyor...")
trainer.train()

# ==========================================
# 3. KAYIT VE DRIVE (OTOMATİK)
# ==========================================
# Drive zaten bağlı.
out_dir = training_args.output_dir
final_dir = os.path.join(out_dir, "FINAL_EXPORT")
os.makedirs(final_dir, exist_ok=True)

# Save Model
best_model_dir = os.path.join(final_dir, "best_model")
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

# Checkpoints
ckpt_export_dir = os.path.join(final_dir, "checkpoints")
if os.path.exists(ckpt_export_dir): shutil.rmtree(ckpt_export_dir)
os.makedirs(ckpt_export_dir, exist_ok=True)
for name in os.listdir(out_dir):
    if name.startswith("checkpoint-"):
        shutil.copytree(os.path.join(out_dir, name), os.path.join(ckpt_export_dir, name))

# Grafik
loss_json = "loss_log_deep.json"
if os.path.exists(loss_json):
    shutil.copy2(loss_json, os.path.join(final_dir, "loss_log.json"))
    try:
        df = pd.read_json(loss_json)
        plt.figure(figsize=(10, 6))
        if "train_loss" in df.columns: plt.plot(df["step"], df["train_loss"], label="Train", alpha=0.5)
        if "val_loss" in df.columns: plt.plot(df["step"], df["val_loss"], label="Valid", linewidth=2)
        if "test_loss" in df.columns: plt.plot(df["step"], df["test_loss"], label="Test", linestyle="--")
        plt.title("Deep Instruction (No-Question) Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(final_dir, "loss_plot.png"))
    except: pass

# ZIP ve Yükle
zip_name = "FINAL_EXPORT.zip"
zip_path = os.path.join(out_dir, zip_name)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(final_dir):
        for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), final_dir))

print(f"🚀 Drive'a yükleniyor: {drive_folder} ...")
shutil.copy2(zip_path, os.path.join(drive_folder, zip_name))
print("✅✅ İŞLEM TAMAM! Drive'a yüklendi.")