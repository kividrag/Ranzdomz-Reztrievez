from datasets import load_dataset

def prepare_input(template, samples, tokenizer):

    messages = [
        [
            {
                "role": "user",
                "content": template.format(**sample),
            }
        ]
        for sample in samples]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to("cuda")

    return input_ids

def load_and_prepare_dataset(dataset_name, tokenizer=None):

    dataset = load_dataset(dataset_name)

    return dataset
