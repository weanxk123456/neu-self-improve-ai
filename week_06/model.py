"""One shared HF model. Only planner calls enable LoRA; other roles are frozen."""
from contextlib import nullcontext
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


class LocalModel:
    def __init__(self, name, context=2048, max_new=256, adapter=None,
                 train=False, four_bit=False, rank=16, alpha=32):
        if context <= max_new + 32:
            raise ValueError('context must exceed max_new + 32')
        if four_bit and not torch.cuda.is_available():
            raise ValueError('4-bit mode requires CUDA and bitsandbytes')
        self.context, self.max_new = context, max_new
        self.training_mode = train
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cuda = torch.cuda.is_available()
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        kwargs = {'torch_dtype': dtype, 'attn_implementation': 'sdpa'}
        if cuda:
            kwargs['device_map'] = {'': torch.cuda.current_device()}
        if four_bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if four_bit and train:
            self.model = prepare_model_for_kbit_training(self.model)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=train)
        elif train:
            self.model = get_peft_model(self.model, LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
        self.has_adapter = isinstance(self.model, PeftModel)
        if train:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False
            self.model.print_trainable_parameters()
        # eval mode disables dropout but still permits gradients during policy updates.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device

    def scope(self, adapter):
        if adapter and not self.has_adapter:
            raise ValueError('Planner adapter requested but none is loaded')
        return self.model.disable_adapter() if self.has_adapter and not adapter else nullcontext()

    def encode(self, system, user):
        # Reserve room for output; retain the system prompt and the start of user input
        # (which includes the original question). Bound evidence in core.Agent as well.
        budget = self.context - self.max_new
        user_ids = self.tokenizer.encode(user, add_special_tokens=False)
        while True:
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': self.tokenizer.decode(user_ids)}]
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if len(ids) <= budget:
                return ids
            remove = len(ids) - budget + 8
            if len(user_ids) <= remove:
                raise ValueError('Context too small for system prompt; increase --context')
            user_ids = user_ids[:-remove]

    @torch.no_grad()
    def generate(self, system, user, adapter=False, temperature=0.0):
        self.model.eval()
        prompt = self.encode(system, user)
        x = torch.tensor([prompt], device=self.device)
        kwargs = {'max_new_tokens': self.max_new, 'do_sample': temperature > 0,
                  'pad_token_id': self.tokenizer.pad_token_id, 'use_cache': True,
                  'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0, 'num_beams': 1}
        if temperature > 0:
            # Match this exact temperature-scaled distribution in log_probs.
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0, typical_p=1.0)
        with self.scope(adapter):
            y = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), **kwargs)
        action = y[0, len(prompt):].tolist()
        return self.tokenizer.decode(action, skip_special_tokens=True).strip(), prompt, action

    def log_probs(self, decision, adapter=True, temperature=1.0):
        """Only generated planner tokens are targets, including any generated EOS."""
        self.model.train(self.training_mode and torch.is_grad_enabled())
        p, a = decision.prompt_ids, decision.action_ids
        if not a:
            raise ValueError('Empty planner completion')
        x = torch.tensor([p + a], device=self.device)
        with self.scope(adapter):
            logits = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        # Logit at position k predicts token k+1. No prompt/tool token enters loss.
        logits = logits[0, len(p)-1:-1, :].float() / temperature
        target = x[0, len(p):]
        return logits.gather(-1, target[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
