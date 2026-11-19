"""Text generation backends with a common interface:

    generate(prompts: list[list[int]], max_new_tokens, temperature, stop) -> list[str]

HFGenerator  : uses the (LoRA) policy being trained directly. Simple, slower.
VLLMGenerator: vLLM engine; the current LoRA adapter is saved to disk and hot-
               loaded each step (LoRARequest). Much faster rollouts.
Sampling is plain (top_p=1, no top_k, no repetition penalty) so that the
behaviour policy matches the policy whose log-probs we train on. Qwen's
generation_config defaults (top_k=20, top_p=0.8, ...) are overridden on purpose.
"""
from .rollout import truncate_at_stop


class HFGenerator:
    def __init__(self, model, tokenizer, batch_size: int = 64):
        self.model, self.tok, self.batch_size = model, tokenizer, batch_size

    def generate(self, prompts, max_new_tokens, temperature, stop=None):
        import torch

        pad = self.tok.pad_token_id
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        outs = []
        try:
            for i in range(0, len(prompts), self.batch_size):
                chunk = prompts[i : i + self.batch_size]
                L = max(len(p) for p in chunk)
                ids = torch.tensor([[pad] * (L - len(p)) + p for p in chunk], device=dev)
                am = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in chunk], device=dev)
                kw = dict(do_sample=False) if temperature == 0 else dict(
                    do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
                with torch.no_grad():
                    gen = self.model.generate(
                        input_ids=ids, attention_mask=am, max_new_tokens=max_new_tokens,
                        pad_token_id=pad, use_cache=True, repetition_penalty=1.0,
                        stop_strings=stop, tokenizer=self.tok if stop else None, **kw)
                for row in gen[:, L:]:
                    text = self.tok.decode(row, skip_special_tokens=True)
                    outs.append(truncate_at_stop(text, stop) if stop else text)
        finally:
            if was_training:
                self.model.train()
        return outs


class VLLMGenerator:
    """Create this BEFORE loading the training models (vLLM reserves a fraction of GPU memory)."""

    def __init__(self, model_name, gpu_memory_utilization=0.3, max_model_len=8192,
                 enable_lora=True, max_lora_rank=64, seed=0):
        from vllm import LLM

        self.llm = LLM(model=model_name, dtype="bfloat16", seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enable_lora=enable_lora, max_lora_rank=max_lora_rank, max_loras=1,
                       enable_prefix_caching=True)
        self.lora_request = None

    def set_adapter(self, path: str, version: int):
        from vllm.lora.request import LoRARequest

        self.lora_request = LoRARequest(f"policy_v{version}", version, path)  # new id => reload

    def generate(self, prompts, max_new_tokens, temperature, stop=None):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sp = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                            max_tokens=max_new_tokens, stop=stop, include_stop_str_in_output=True)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                                 lora_request=self.lora_request, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
