# model_utils.py
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


print("[INFO] Loading model...")
device = get_device()
print(f"[INFO] Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Для CPU грузим обычно, для GPU можно auto + float16
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)

model.eval()
print("[INFO] Model loaded successfully.")


def build_prompt(user_message: str, history: list[dict] | None = None) -> str:
    """Формирует промпт в chat-template формате Qwen Instruct."""
    messages = []

    if history:
        for item in history[-5:]:
            user_text = item.get("user", "")
            bot_text = item.get("bot", "")
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if bot_text:
                messages.append({"role": "assistant", "content": bot_text})

    messages.append({"role": "user", "content": user_message})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_answer(user_message: str, history: list[dict] | None = None) -> dict:
    """Генерирует ответ и возвращает текст + latency + throughput."""
    prompt = build_prompt(user_message, history)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    end_time = time.time()

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    latency = end_time - start_time
    token_count = len(new_tokens)
    throughput = token_count / latency if latency > 0 else 0

    return {
        "response": response,
        "latency": latency,
        "tokens": token_count,
        "throughput": throughput,
    }
