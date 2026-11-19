"""Policy (actor), reference model and critic, all parameter-efficient (LoRA).

* Actor  : Qwen2.5-3B-Instruct + LoRA adapter (only the adapter is trained).
* Ref    : the SAME model with the adapter disabled (`with policy.disable_adapter()`),
           i.e. the frozen initial policy -> no second copy of the weights.
* Critic : (PPO only) a second Qwen backbone + its own LoRA adapter + a linear
           value head (initialised to zero).
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _attn_impl():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def lora_config(r, alpha, dropout, task_type):
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                      target_modules=TARGET_MODULES, bias="none", task_type=task_type)


def load_policy(model_name, r=32, alpha=64, dropout=0.0, grad_ckpt=True, device="cuda"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=_attn_impl()).to(device)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(r, alpha, dropout, "CAUSAL_LM"))
    model.print_trainable_parameters()
    return model


class Critic(nn.Module):
    def __init__(self, model_name, r=32, alpha=64, dropout=0.0, grad_ckpt=True, device="cuda"):
        super().__init__()
        from transformers import AutoModel

        base = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation=_attn_impl()).to(device)
        if grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        self.backbone = get_peft_model(base, lora_config(r, alpha, dropout, "FEATURE_EXTRACTION"))
        self.value_head = nn.Linear(base.config.hidden_size, 1).to(device)  # fp32 head
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask, position_ids, resp_len: int):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=position_ids, use_cache=False).last_hidden_state
        h = h[:, -resp_len - 1 : -1]          # state before each response token (same slice as log-probs)
        return self.value_head(h.float()).squeeze(-1)
