
# 🤖 ZETTA Chatbot with Fine-tuned GPT-2 Medium (DialoGPT)

ZETTA is a persistent chatbot powered by a fine-tuned DialoGPT-medium model. It maintains multi-turn conversation history across browser refreshes using `session_id` and stores memory on disk for persistent context.

---

## 🚀 Features

- Fine-tuned `DialoGPT-medium` using custom dataset
- Chat memory saved per user session (`session_id`)
- Memory is persistent (saved in `chat_memory.json`)
- Frontend: modern HTML/CSS interface with automatic session handling
- Backend: Flask API with PyTorch + Transformers
- Auto-detects GPU with `torch.cuda.is_available()`

---

## 🗂️ Project Structure

```
├── app.py                  # Flask backend with persistent memory
├── finetune_gpt2.py        # Fine-tuning script (DialoGPT)
├── prepare_dataset.py      # Format chat dataset into training format
├── chat_data.txt           # Original chat dataset
├── dialogs.txt             # Cleaned and formatted conversation pairs
├── formatted_dataset.txt   # Final file used for training
├── geni_model/             # Fine-tuned DialoGPT-medium model
├── templates/
│   └── index.html          # Frontend chat interface
├── static/                 # (Optional) CSS or JS if separated
└── requirements.txt
```

---

## ⚙️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
transformers
datasets
torch
flask
flask-cors
```

---

## 🧠 Fine-tuning

Prepare the dataset:

```bash
python prepare_dataset.py
```

Start fine-tuning:

```bash
python finetune_gpt2.py
```

---

## 🖥️ Run the Backend

```bash
python app.py
```

Flask server runs at `http://127.0.0.1:5000`.

---

## 🌐 Frontend

Simply open the HTML file at `templates/index.html` in a browser.

It uses `localStorage` to maintain unique session IDs for consistent chat history.

---

## 📂 Persistent Chat History

- Chat histories are stored in `chat_memory.json`
- Memory is maintained between page refreshes or backend restarts

---

## 🧪 Example

You say:
```
Hi, who are you?
```

ZETTA replies:
```
Hello! I’m ZETTA, your AI assistant. How can I help today?
```

---

## 📜 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue for suggestions or bug reports.
