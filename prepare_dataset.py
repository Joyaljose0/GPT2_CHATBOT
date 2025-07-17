def convert_to_dialogpt_format(input_file, output_file, max_tokens=1024):
    from transformers import AutoTokenizer

    # Load DialoGPT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token

    formatted_conversations = []
    conversation = []

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("User:"):
            user_msg = line[len("User:"):].strip()
            conversation.append(user_msg)

        elif line.startswith("Bot:"):
            bot_msg = line[len("Bot:"):].strip()
            conversation.append(bot_msg)

            # Once we have a User+Bot pair, prepare a full example
            joined = "<|endoftext|>".join(conversation) + "<|endoftext|>"
            tokenized = tokenizer(joined, truncation=True, max_length=max_tokens)

            if len(tokenized["input_ids"]) <= max_tokens:
                formatted_conversations.append(joined + "\n")

            conversation = []  # Reset after each pair

    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(formatted_conversations)

    print(f"✅ DialoGPT formatted dataset saved to {output_file}")


if __name__ == "__main__":
    convert_to_dialogpt_format("chat_data.txt", "formatted_dataset.txt")
