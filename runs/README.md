# Runs

Каждая папка `vN_*/` - отдельный эксперимент.

Структура папки:
- `train_notebook.py` - версия скрипта, которой обучали
- `config.json` - гиперпараметры (скачивается из Kaggle output)
- `eval.json` - метрики на holdout (rougeL_f_mean - главная)
- `predictions.jsonl` - предсказания с reference и ROUGE
- `merged.tar` - упакованные веса (опционально, ~3.8GB)

`best/` - симлинк/копия лучшего варианта, веса оттуда → корневой `weights/` для submission.
