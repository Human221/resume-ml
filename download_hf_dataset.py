"""
Скрипт для загрузки и проверки датасета с Hugging Face
"""

import argparse
import json
import requests
from datasets import load_dataset


def list_splits(dataset_name: str):
    """Получить список доступных splits для датасета"""
    url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_name.replace('/', '%2F')}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        splits = data.get('splits', [])
        print(f"\nДоступные splits для {dataset_name}:")
        for split in splits:
            print(f"  - {split.get('split', 'unknown')}: {split.get('num_examples', 'unknown')} примеров")
        
        return splits
    except Exception as e:
        print(f"Ошибка при получении splits: {e}")
        return []


def download_sample(dataset_name: str, split: str = "train", num_samples: int = 10):
    """Загрузить образцы данных через API"""
    base_url = "https://datasets-server.huggingface.co/rows"
    url = f"{base_url}?dataset={dataset_name.replace('/', '%2F')}&config=default&split={split}&offset=0&length={num_samples}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        rows = data.get('rows', [])
        print(f"\nЗагружено {len(rows)} примеров:")
        print("=" * 70)
        
        for i, row in enumerate(rows, 1):
            row_data = row.get('row', {})
            print(f"\nПример {i}:")
            print(json.dumps(row_data, ensure_ascii=False, indent=2)[:500])
            if len(json.dumps(row_data, ensure_ascii=False)) > 500:
                print("...")
        
        return rows
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return []


def load_full_dataset(dataset_name: str, split: str = "train", max_samples: int = None):
    """Загрузить полный датасет через библиотеку datasets"""
    print(f"\nЗагрузка датасета {dataset_name} (split: {split})...")
    
    try:
        if max_samples:
            dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        print(f"Загружено {len(dataset)} примеров")
        print(f"\nСтруктура данных:")
        print(dataset)
        
        if len(dataset) > 0:
            print(f"\nПервый пример:")
            print(json.dumps(dataset[0], ensure_ascii=False, indent=2)[:500])
            if len(json.dumps(dataset[0], ensure_ascii=False)) > 500:
                print("...")
        
        return dataset
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Загрузка и проверка датасета с Hugging Face")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evilfreelancer/headhunter",
        help="Название датасета на Hugging Face"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split для загрузки"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["list", "sample", "load"],
        default="list",
        help="Действие: list (список splits), sample (образцы), load (полная загрузка)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Количество образцов для загрузки"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ЗАГРУЗКА ДАТАСЕТА С HUGGING FACE")
    print("=" * 70)
    print(f"Датасет: {args.dataset}")
    print(f"Split: {args.split}")
    print("=" * 70)
    
    if args.action == "list":
        list_splits(args.dataset)
    elif args.action == "sample":
        download_sample(args.dataset, args.split, args.num_samples)
    elif args.action == "load":
        load_full_dataset(args.dataset, args.split, args.num_samples)


if __name__ == "__main__":
    main()

