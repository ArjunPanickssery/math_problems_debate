import functools
import os
# from transformers import BitsAndBytesConfig

@functools.cache
def load_hf_model(model: str, hf_quantization_config=True):
    print("Loading Hugging Face model", model, hf_quantization_config)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = (
        BitsAndBytesConfig(load_in_8bit=True) if hf_quantization_config else None
    )

    api_key = os.getenv("HF_TOKEN")
    model = model.split("hf:")[1]
    tokenizer = AutoTokenizer.from_pretrained(model)
    device_map = "cuda" if hf_quantization_config else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=device_map,
        token=api_key,
        quantization_config=quant_config,
    )

    return (tokenizer, model)