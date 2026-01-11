from unsloth import FastLanguageModel

def apply_lora(model):

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model


def load_and_prepare_model(model_name="LiquidAI/LFM2.5-1.2B-Instruct", peft_method="none"):

    # Load model in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,  # Enable 4-bit quantization
    )

    if peft_method=="LoRA":
        model = apply_lora(model)

    return model, tokenizer
