
# 🤖 ZETTA Chatbot with Fine-tuned GPT-2 Medium 

ZETTA is a personalized chatbot powered by a fine-tuned version of OpenAI’s GPT-2 Medium model. Designed to be friendly, helpful, and memory-aware, ZETTA can remember past conversations—even after a server restart—thanks to persistent session memory stored on disk. This makes it ideal for building intelligent assistants, learning bots, and conversational agents that require context continuity.

Whether you're experimenting with dialogue generation or deploying your own AI assistant, ZETTA is built for developers who want flexibility, extensibility, and a clean chat interface out of the box.

---

## 🚀 Features

- Fine-tuned `GPT2-medium` using custom dataset
- Chat memory saved per user session (`session_id`)
- Memory is persistent (saved in `chat_memory.json`)
- Frontend: modern HTML/CSS interface with automatic session handling
- Backend: Flask API with PyTorch + Transformers
- Auto-detects GPU with `torch.cuda.is_available()`

---

## 🗂️ Project Structure

```
├── app.py                  # Flask backend with persistent memory
├── finetune_gpt2.py        # Fine-tuning script (GPT2 - Medium)
├── prepare_dataset.py      # Format chat dataset into training format
├── chat_data.txt           # Original chat dataset
├── dialogs.txt             # Cleaned and formatted conversation pairs
├── formatted_dataset.txt   # Final file used for training
├── geni_model/             # Fine-tuned GPT2-medium model
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
