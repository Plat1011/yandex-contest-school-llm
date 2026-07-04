# Yandex Contest - Effective Inference on School Questions

Дистилляция "умной" LLM в маленькую для контеста "Ответы на школьные вопросы".
Судья контеста - закрытая reward-модель. Инференс: L4 24 GB, ≤ 15 мин на 4000 запросов, docker-образ ≤ 20 GB, submission zip ≤ 10 GB.

Baseline (Qwen3-0.6B + vLLM) задает нижнюю планку. Цель - набрать максимум.

## Итог

| Версия | Модель / SFT | ROUGE-L (holdout 200) | Балл судьи |
|---|---|---|---|
| Baseline | Qwen3-0.6B, no SFT | - | - |
| Qwen3-1.7B raw | Qwen3-1.7B, no SFT, `max_tokens=512` | - | 56 |
| Qwen3-1.7B raw | Qwen3-1.7B, no SFT, `max_tokens=1024` | - | 51.5 |
| QVikhr-3-1.7B-Instruction | русско-натюненный Qwen3-1.7B | - | 48.2 |
| v2 | Qwen3-1.7B + LoRA `r=32, e=2`, `max_tokens=512` | - | 56 |
| v3 | Qwen3-1.7B + LoRA `r=32, e=2`, `max_tokens=512` | 0.526 | 56 |
| v5 | Qwen3-1.7B + LoRA `r=128, e=3`, `max_tokens=512` | 0.529 | 58.1 |
| v6 | Qwen3-1.7B + LoRA `r=256, e=3`, `max_tokens=512` | 0.542 | 57.9 |

Лучший результат: 58.1 (v5, Qwen3-1.7B + aggressive LoRA-SFT).

## Выводы

1. Русско-специфический fine-tune Vikhr ухудшил балл. QVikhr-3-1.7B (Ru Arena General 59.2 против Qwen3-1.7B 49.7 в паблике) выдал 48.2 балла судьи - стиль тренинга Vikhr не совпадает со стилем reference-ответов контеста.
2. SFT работает, но только достаточно агрессивный. r=32 (v3) не сдвинул балл, r=128 (v5) добавил +2.1. Ранг 256 (v6) продолжил тренд по ROUGE (0.542 vs 0.529), но у судьи 57.9 - имеет место насыщение.
3. Длина ответа критична. Расширение `max_tokens` с 512 до 1024 на raw Qwen3-1.7B снизило балл (51.5). Модель без SFT начинает "галлюцинировать" на длинных генерациях; после SFT такая проблема исчезает.
4. Reference-ответы в train.parquet не задают полностью "правильный" стиль. SFT на reference не дает большого прироста - судья, видимо, оценивает content-correctness, а не форматирование.
5. Внутренняя ROUGE-L не коррелирует с судьей. v3 (0.526) и v5 (0.529) отличаются на 0.003, при этом балл судьи вырос на +2.1. Мелкие ROUGE-сдвиги не предсказывают судью.

## Структура репозитория

```
├── Dockerfile
├── solution.py
├── download_weights.py
├── train.py
├── pack_submission.py
├── make_viewer.py
├── viewer.html
├── dataset_ml_challenge.parquet
├── example-dataset.pickle
├── kaggle_train/
├── kaggle_fork/
├── kaggle_dataset/
└── runs/
    ├── README.md
    ├── BEST/
    ├── v1_.../
    ├── v3_qwen3_1.7b_r32_hfgen/
    ├── v4_qwen3_1.7b_r64/
    ├── v5_qwen3_1.7b_r128_e3/
    └── v6_qwen3_1.7b_r256_e3/
```

Файлы весов (`weights/`, `*.tar`, `*.safetensors`, `submission*.zip`) в `.gitignore` - крупные бинарники.

## Хронология

Каждая итерация - отдельный SFT-прогон на Kaggle T4:

- v1 - исходный ноутбук, no eval. Упал (transformers 4.46 не поддерживал Qwen3).
- v2 - добавлен holdout+ROUGE eval. Все еще transformers 4.46 → упал. Веса тренинга удалось извлечь с диска Kaggle, но без метрик.
- v3 - transformers 4.51 + fp16 + eval через `transformers.generate` (без vLLM ABI-проблем). ROUGE-L 0.526. Балл судьи 56.
- v4 - планировался `r=64`, не запускался.
- v5 - агрессивный SFT: `r=128, alpha=256, epochs=3, lr=1e-4`, `batch=1, grad_accum=16`. Обучение 4.7ч на T4. ROUGE-L 0.529. Балл 58.1 - лучший.
- v6 - `r=256, alpha=512, epochs=3, lr=8e-5`. Обучение ~5ч. ROUGE-L 0.542 (лучший по ROUGE). Балл 57.9.
- QVikhr-3-1.7B-Instruction без SFT - 48.2 балла.

## Оценка результатов

`make_viewer.py` собирает предсказания всех обученных версий по 200 holdout-вопросам в интерактивный HTML - reference и все версии бок-о-бок, с фильтрами по тексту вопроса и ROUGE-L.

```bash
python make_viewer.py
```

## Ограничения контеста

- L4 24 GB VRAM, 15 GB RAM, 8 CPU cores.
- `≤ 15` минут на 4000 вопросов.
- Образ `≤ 20` GB, submission zip `≤ 10` GB.
- Нет интернета в контейнере - все веса и пакеты внутри.
- 10 попыток в сутки, 40 суммарно.
