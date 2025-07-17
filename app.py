from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import json

app = Flask(__name__)
CORS(app)

#  Path to the fine-tuned GPT-2 Medium model
MODEL_DIR = "./geni_model"

#  Ensure all necessary model files exist
required_files = [
    "config.json", "pytorch_model.bin", "vocab.json",
    "merges.txt", "tokenizer_config.json", "special_tokens_map.json"
]
missing = [f for f in required_files if not os.path.isfile(os.path.join(MODEL_DIR, f))]
if missing:
    raise FileNotFoundError(f"❌ Missing files in '{MODEL_DIR}': {missing}")

#  Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

#  Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

#  Memory storage file
MEMORY_FILE = "chat_memory.json"

#  Load memory if exists
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        chat_memory = json.load(f)
else:
    chat_memory = {}

#  Convert JSON memory to tensor format
def get_history(session_id):
    if session_id in chat_memory:
        return torch.tensor(chat_memory[session_id], dtype=torch.long).unsqueeze(0)
    return None

#  Save tensor history to disk
def save_history(session_id, history_tensor):
    chat_memory[session_id] = history_tensor[0].tolist()
    with open(MEMORY_FILE, "w") as f:
        json.dump(chat_memory, f)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    session_id = data.get("session_id", "").strip()

    if not user_input:
        return jsonify({"reply": "Please enter a message."})
    if not session_id:
        return jsonify({"reply": "Missing session ID."})

    #  Load memory for this session
    chat_history_ids = get_history(session_id)

    # Format input
    formatted_input = f"<|user|> {user_input} <|bot|>"
    new_input_ids = tokenizer.encode(formatted_input + tokenizer.eos_token, return_tensors="pt")

    # Append to history
    input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    #  Generate response
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        temperature=0.85,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract bot response
    reply_ids = chat_history_ids[:, input_ids.shape[-1]:]
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()

    #  Save history
    if chat_history_ids.shape[-1] > 900:
        chat_history_ids = None
        chat_memory.pop(session_id, None)  # Optional: clear session if too long
    else:
        save_history(session_id, chat_history_ids)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    print(f" Running fine-tuned GPT-2 Medium chatbot from: {MODEL_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=True)
