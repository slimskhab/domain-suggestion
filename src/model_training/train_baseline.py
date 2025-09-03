"""
Model Training Module for Domain Name Generation

This module handles the training of fine-tuned models for domain name suggestions.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from datetime import datetime

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HuggingFaceDataset
import wandb

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    base_model: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    save_steps: int
    eval_steps: int
    logging_steps: int
    output_dir: str

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]

class DomainNameTrainer:
    """Trainer class for domain name generation models."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the trainer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model_and_tokenizer(self, model_name: str = None):
        """Load the base model and tokenizer."""
        if model_name is None:
            model_name = self.config['model']['base_model']
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"✅ Model and tokenizer loaded")
        
    def prepare_dataset(self, df: pd.DataFrame) -> HuggingFaceDataset:
        """Prepare dataset for training."""
        def tokenize_function(examples):
            # Combine input and target with separator
            combined_texts = [
                f"{input_text} -> {target_text}" 
                for input_text, target_text in zip(examples['input_text'], examples['target_text'])
            ]
            
            # Tokenize
            tokenized = self.tokenizer(
                combined_texts,
                truncation=True,
                padding=True,
                max_length=self.config['dataset']['max_input_length'] + self.config['dataset']['max_output_length'],
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Convert to HuggingFace Dataset
        dataset = HuggingFaceDataset.from_pandas(df)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['model']['lora']['r'],
            lora_alpha=self.config['model']['lora']['lora_alpha'],
            lora_dropout=self.config['model']['lora']['lora_dropout'],
            target_modules=self.config['model']['lora']['target_modules']
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ LoRA configuration applied")
    
    def setup_training_args(self, output_dir: str = None) -> TrainingArguments:
        """Setup training arguments."""
        if output_dir is None:
            output_dir = f"./models/baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['model']['training']['num_epochs'],
            per_device_train_batch_size=self.config['model']['training']['batch_size'],
            per_device_eval_batch_size=self.config['model']['training']['batch_size'],
            gradient_accumulation_steps=self.config['model']['training']['gradient_accumulation_steps'],
            learning_rate=self.config['model']['training']['learning_rate'],
            warmup_steps=self.config['model']['training']['warmup_steps'],
            weight_decay=self.config['model']['training']['weight_decay'],
            logging_steps=self.config['model']['training']['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['model']['training']['eval_steps'],
            save_steps=self.config['model']['training']['save_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if os.getenv('WANDB_API_KEY') else None,
            save_total_limit=3,
            dataloader_pin_memory=False,
        )
        
        return training_args
    
    def train_baseline_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           model_name: str = None, output_dir: str = None):
        """Train the baseline model."""
        print("🚀 Starting baseline model training...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer(model_name)
        
        # Setup LoRA
        self.setup_lora()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Setup training arguments
        training_args = self.setup_training_args(output_dir)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Train the model
        self.trainer.train()
        
        # Save the model
        self.trainer.save_model(f"{training_args.output_dir}/final")
        self.tokenizer.save_pretrained(f"{training_args.output_dir}/final")
        
        print(f"✅ Baseline model training complete! Saved to {training_args.output_dir}/final")
        
        return training_args.output_dir
    
    def train_improved_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           base_model_path: str, output_dir: str = None):
        """Train an improved model based on previous iterations."""
        print("🚀 Starting improved model training...")
        
        # Load the base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Setup LoRA (can be different from baseline)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Increased rank
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Setup training arguments with different hyperparameters
        training_args = TrainingArguments(
            output_dir=output_dir or f"./models/improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            num_train_epochs=2,  # Fewer epochs for fine-tuning
            per_device_train_batch_size=4,  # Smaller batch size
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,  # Lower learning rate
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if os.getenv('WANDB_API_KEY') else None,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Train the model
        self.trainer.train()
        
        # Save the model
        self.trainer.save_model(f"{training_args.output_dir}/final")
        self.tokenizer.save_pretrained(f"{training_args.output_dir}/final")
        
        print(f"✅ Improved model training complete! Saved to {training_args.output_dir}/final")
        
        return training_args.output_dir
    
    def generate_domain_names(self, prompt: str, num_sequences: int = 5) -> List[str]:
        """Generate domain names using the trained model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        generation_config = GenerationConfig(
            max_length=self.config['model']['generation']['max_length'],
            temperature=self.config['model']['generation']['temperature'],
            top_p=self.config['model']['generation']['top_p'],
            do_sample=self.config['model']['generation']['do_sample'],
            num_return_sequences=num_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Extract the generated part (after the prompt)
            if " -> " in text:
                generated_part = text.split(" -> ")[-1]
                generated_texts.append(generated_part.strip())
            else:
                generated_texts.append(text.strip())
        
        return generated_texts

def main():
    """Main function to train the baseline model."""
    print("🚀 Starting domain name model training...")
    
    # Load dataset
    df = pd.read_csv("data/synthetic_dataset/domain_names_dataset.csv")
    
    # Split dataset
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize trainer
    trainer = DomainNameTrainer()
    
    # Train baseline model
    model_path = trainer.train_baseline_model(train_df, val_df)
    
    print(f"✅ Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()
