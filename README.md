# Alfa

Решение задачи бинарного ранжирования отклика на кредитный оффер. Модель строит score вероятности отклика клиента, а финальное ранжирование оптимизируется под ROC-AUC.

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_credit_offer_model.py --data-dir data --output-dir outputs
```

После запуска основные артефакты появятся в `outputs/`: `submission.csv`, `validation_report.json`, варианты сабмитов и важности признаков.

## Дополнительные варианты

```bash
python make_public_variants.py --data-dir data --output-dir outputs --output-name submission_full_cat_heavy_rank.csv --prediction-name final_model_test_predictions_full_cat_heavy.csv
```

Опциональные флаги:

- `--train-start-date YYYY-MM-DD` - обучать вариант только на данных с указанной даты.
- `--drop-raw-date` - убрать сырые календарные признаки.
- `--drop-month-context` - убрать month-context признаки.
