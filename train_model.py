"""
Скрипт для обучения модели на Cloud.ru
Поддерживает fine-tuning моделей для анализа вакансий
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()


class VacancyDataset:
    """Класс для подготовки данных вакансий для обучения"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """Загрузка данных из CSV"""
        print(f"Загрузка данных из {self.csv_path}...")
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
            print(f"Загружено {len(self.data)} записей")
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            self.data = []
    
    def prepare_training_data(self, max_samples: int = None) -> List[str]:
        """Подготовка данных для обучения в формате промптов"""
        training_texts = []
        
        samples = self.data[:max_samples] if max_samples else self.data
        
        for vacancy in samples:
            # Формируем текст для обучения
            name = vacancy.get('Name', '')
            description = vacancy.get('Description', '')
            salary_from = vacancy.get('From', '')
            salary_to = vacancy.get('To', '')
            experience = vacancy.get('Experience', '')
            area = vacancy.get('Area', '')
            roles = vacancy.get('Professional roles', '')
            
            # Создаем промпт в формате для модели
            prompt = f"""<|im_start|>system
Ты - HR-ассистент, специализирующийся на анализе вакансий.<|im_end|>
<|im_start|>user
Опиши вакансию: {name}<|im_end|>
<|im_start|>assistant
Название: {name}
Описание: {description[:500]}
Зарплата: {salary_from} - {salary_to} рублей
Опыт: {experience}
Регион: {area}
Роль: {roles}<|im_end|>
"""
            training_texts.append(prompt)
        
        return training_texts


def prepare_dataset(texts: List[str], tokenizer, max_length: int = 512):
    """Подготовка датасета для обучения"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset


def train_model(
    model_name: str = "IlyaGusev/saiga_mistral_7b_merged",
    csv_path: str = "IT_vacancies_full 2.csv",
    output_dir: str = "./models/finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_samples: int = 1000,
    max_length: int = 512
):
    """Обучение модели"""
    
    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ АНАЛИЗА ВАКАНСИЙ")
    print("=" * 70)
    print(f"Модель: {model_name}")
    print(f"Данные: {csv_path}")
    print(f"Выходная директория: {output_dir}")
    print(f"Эпохи: {num_epochs}")
    print(f"Размер батча: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 70)
    print()
    
    # Проверка доступности GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Загрузка токенизатора и модели
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Установка pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Подготовка данных
    print("\nПодготовка данных для обучения...")
    dataset = VacancyDataset(csv_path)
    training_texts = dataset.prepare_training_data(max_samples=max_samples)
    
    if not training_texts:
        print("Ошибка: нет данных для обучения!")
        return
    
    print(f"Подготовлено {len(training_texts)} примеров для обучения")
    
    # Токенизация данных
    print("\nТокенизация данных...")
    tokenized_dataset = prepare_dataset(training_texts, tokenizer, max_length=max_length)
    
    # Разделение на train/val
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    print(f"Train: {len(train_dataset)} примеров")
    print(f"Val: {len(val_dataset)} примеров")
    
    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",
        gradient_accumulation_steps=4,
        report_to="none",
        save_total_limit=3,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Обучение
    print("\n" + "=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 70)
    print()
    
    trainer.train()
    
    # Сохранение модели
    print("\nСохранение модели...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Модель сохранена в: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Обучение модели для анализа вакансий")
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "IlyaGusev/saiga_mistral_7b_merged"),
        help="Название базовой модели"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.getenv("VACANCIES_CSV_PATH", "IT_vacancies_full 2.csv"),
        help="Путь к CSV файлу с вакансиями"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/finetuned",
        help="Директория для сохранения обученной модели"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Количество эпох обучения"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Размер батча"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Максимальное количество примеров для обучения"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Максимальная длина последовательности"
    )
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model_name,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()

