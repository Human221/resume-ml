"""
Скрипт для обучения модели на Cloud.ru
Поддерживает fine-tuning моделей для анализа вакансий
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import requests
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv()


class HuggingFaceDatasetLoader:
    """Класс для загрузки данных с Hugging Face через API"""
    
    def __init__(self, dataset_name: str = "evilfreelancer/headhunter", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
    
    def load_from_api(self, max_samples: Optional[int] = None, batch_size: int = 100):
        """Загрузка данных через Hugging Face Datasets Server API"""
        print(f"Загрузка данных с Hugging Face: {self.dataset_name} (split: {self.split})...")
        
        try:
            # Используем библиотеку datasets для загрузки
            print("Загрузка через datasets.load_dataset...")
            dataset = load_dataset(self.dataset_name, split=self.split)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Конвертируем в список словарей
            self.data = [item for item in dataset]
            print(f"Загружено {len(self.data)} записей")
            
        except Exception as e:
            print(f"Ошибка загрузки через datasets: {e}")
            print("Попытка загрузки через API...")
            self._load_via_api(max_samples, batch_size)
    
    def _load_via_api(self, max_samples: Optional[int] = None, batch_size: int = 100):
        """Загрузка данных через прямой API запрос"""
        base_url = "https://datasets-server.huggingface.co/rows"
        offset = 0
        total_loaded = 0
        
        while True:
            if max_samples and total_loaded >= max_samples:
                break
            
            length = batch_size
            if max_samples:
                length = min(batch_size, max_samples - total_loaded)
            
            url = f"{base_url}?dataset={self.dataset_name.replace('/', '%2F')}&config=default&split={self.split}&offset={offset}&length={length}"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                rows = data.get('rows', [])
                if not rows:
                    break
                
                for row in rows:
                    self.data.append(row.get('row', {}))
                    total_loaded += 1
                    
                    if max_samples and total_loaded >= max_samples:
                        break
                
                offset += length
                
                if len(rows) < length:
                    break
                    
                print(f"Загружено {total_loaded} записей...", end='\r')
                
            except Exception as e:
                print(f"\nОшибка при загрузке данных: {e}")
                break
        
        print(f"\nВсего загружено {len(self.data)} записей")
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Получить загруженные данные"""
        return self.data


class VacancyDataset:
    """Класс для подготовки данных вакансий для обучения"""
    
    def __init__(self, csv_path: Optional[str] = None, hf_dataset: Optional[str] = None, hf_split: str = "train"):
        self.csv_path = csv_path
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """Загрузка данных из CSV или Hugging Face"""
        if self.hf_dataset:
            # Загрузка с Hugging Face
            loader = HuggingFaceDatasetLoader(self.hf_dataset, self.hf_split)
            loader.load_from_api()
            self.data = loader.get_data()
        elif self.csv_path:
            # Загрузка из CSV
            print(f"Загрузка данных из {self.csv_path}...")
            try:
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.data = list(reader)
                print(f"Загружено {len(self.data)} записей")
            except Exception as e:
                print(f"Ошибка загрузки данных: {e}")
                self.data = []
        else:
            raise ValueError("Необходимо указать либо csv_path, либо hf_dataset")
    
    def prepare_training_data(self, max_samples: int = None) -> List[str]:
        """Подготовка данных для обучения в формате промптов"""
        training_texts = []
        
        samples = self.data[:max_samples] if max_samples else self.data
        
        for vacancy in samples:
            # Поддержка разных форматов данных (CSV и Hugging Face)
            # Пробуем разные варианты названий полей
            name = vacancy.get('Name', '') or vacancy.get('name', '') or vacancy.get('title', '')
            description = vacancy.get('Description', '') or vacancy.get('description', '') or vacancy.get('desc', '')
            salary_from = vacancy.get('From', '') or vacancy.get('from', '') or vacancy.get('salary_from', '')
            salary_to = vacancy.get('To', '') or vacancy.get('to', '') or vacancy.get('salary_to', '')
            experience = vacancy.get('Experience', '') or vacancy.get('experience', '') or vacancy.get('exp', '')
            area = vacancy.get('Area', '') or vacancy.get('area', '') or vacancy.get('location', '')
            roles = vacancy.get('Professional roles', '') or vacancy.get('professional_roles', '') or vacancy.get('roles', '')
            
            # Если description - это список или словарь, конвертируем в строку
            if isinstance(description, (list, dict)):
                description = json.dumps(description, ensure_ascii=False)
            
            # Создаем промпт в формате для модели
            prompt = f"""<|im_start|>system
Ты - HR-ассистент, специализирующийся на анализе вакансий.<|im_end|>
<|im_start|>user
Опиши вакансию: {name}<|im_end|>
<|im_start|>assistant
Название: {name}
Описание: {str(description)[:500]}
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
    csv_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    output_dir: str = "./models/finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_samples: Optional[int] = None,
    max_length: int = 512
):
    """Обучение модели"""
    
    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ АНАЛИЗА ВАКАНСИЙ")
    print("=" * 70)
    print(f"Модель: {model_name}")
    if hf_dataset:
        print(f"Данные: Hugging Face - {hf_dataset} (split: {hf_split})")
    else:
        print(f"Данные: CSV - {csv_path}")
    print(f"Выходная директория: {output_dir}")
    print(f"Эпохи: {num_epochs}")
    print(f"Размер батча: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    if max_samples:
        print(f"Максимум примеров: {max_samples}")
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
    dataset = VacancyDataset(
        csv_path=csv_path,
        hf_dataset=hf_dataset,
        hf_split=hf_split
    )
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
        default=os.getenv("VACANCIES_CSV_PATH", None),
        help="Путь к CSV файлу с вакансиями"
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=os.getenv("HF_DATASET", "evilfreelancer/headhunter"),
        help="Название датасета на Hugging Face (например: evilfreelancer/headhunter)"
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Split датасета для загрузки (по умолчанию: train)"
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Использовать Hugging Face датасет вместо CSV"
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
    
    # Определяем источник данных
    csv_path = args.csv_path if not args.use_hf else None
    hf_dataset = args.hf_dataset if args.use_hf else None
    
    # Если не указан ни CSV, ни HF, используем HF по умолчанию
    if not csv_path and not hf_dataset:
        hf_dataset = args.hf_dataset
        print("Используется Hugging Face датасет по умолчанию")
    
    train_model(
        model_name=args.model_name,
        csv_path=csv_path,
        hf_dataset=hf_dataset,
        hf_split=args.hf_split,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()

