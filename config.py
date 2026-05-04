# config.py
# Путь к папке со слитой моделью из ЛР2:
# my_finetuned_model_lab3/
# ├── config.json
# ├── model.safetensors
# ├── tokenizer.json
# └── ...

MODEL_PATH = "./my_finetuned_model_lab3"
MAX_NEW_TOKENS = 180
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.15
