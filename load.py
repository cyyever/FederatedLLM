from transformers import LlamaForCausalLM
import torch

model = LlamaForCausalLM.from_pretrained(
    "tinyllama",
    load_in_8bit=False,
    torch_dtype=torch.float32,
    token="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",
)
