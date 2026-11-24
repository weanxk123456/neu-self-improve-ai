"""Offline neural integration: create a RANDOM tiny Qwen model, exercise real PEFT.
This is a software test, NOT a benchmark or a meaningful language model result.
Run after installing requirements: python tests/integration_tiny.py
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from core import Decision, advantages
from model import LocalModel


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vocab = {'<unk>': 0, '<pad>': 1, '<eos>': 2, 'user': 3, 'assistant': 4,
                 'system': 5, 'hello': 6, 'answer': 7, 'search': 8, 'query': 9}
        tok = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
        tok.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='<unk>',
                    pad_token='<pad>', eos_token='<eos>')
        tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' '}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(root / 'base')
        cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                          num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                          max_position_embeddings=512, eos_token_id=2, pad_token_id=1)
        Qwen2ForCausalLM(cfg).save_pretrained(root / 'base')
        m = LocalModel(str(root/'base'), context=128, max_new=8, train=True, rank=2, alpha=4)
        raw, p, a = m.generate('hello', 'answer', adapter=True, temperature=.5)
        d = Decision(p, a, raw, {})
        assert len(m.log_probs(d, True, .5)) == len(a)
        # Verify the causal target slice against an independently indexed calculation.
        with torch.no_grad():
            x = torch.tensor([p+a], device=m.device)
            logits = m.model(input_ids=x, use_cache=False).logits[0].float() / .5
            expected = torch.stack([logits[len(p)-1+i].log_softmax(-1)[token] for i, token in enumerate(a)])
            old = m.log_probs(d, True, .5).detach()
            ref = m.log_probs(d, False, .5).detach()
            assert torch.allclose(old, expected, atol=1e-6)
        params = [x for x in m.model.parameters() if x.requires_grad]
        before = [x.detach().clone() for x in params]
        opt = torch.optim.AdamW(params, lr=.01)
        new = m.log_probs(d, True, .5)
        ratio = (new-old).exp()
        delta = ref-new
        loss = (-torch.minimum(ratio, ratio.clamp(.8, 1.2)) + .001*(delta.exp()-delta-1)).mean()
        loss.backward()
        assert all(x.grad is not None for x in params)
        opt.step()
        assert any(not torch.equal(b, x) for b, x in zip(before, params))
        with torch.no_grad():
            frozen = m.log_probs(d, False, .5)
            assert torch.allclose(frozen, ref, atol=1e-6), 'Base changed!'
            trained = m.log_probs(d, True, .5).clone()
        m.save(root/'adapter')
        reloaded = LocalModel(str(root/'base'), 128, 8, adapter=str(root/'adapter'))
        with torch.no_grad():
            assert torch.allclose(reloaded.log_probs(d, True, .5), trained, atol=1e-6)
        # Exercise the real trainer with controlled environment outcomes.
        import argparse
        import json
        from unittest.mock import patch
        from core import Trajectory
        from train import train
        (root/'train.jsonl').write_text(json.dumps({'question':'train only', 'answers':['yes']})+'\n')
        (root/'test.jsonl').write_text(json.dumps({'question':'held out', 'answers':['no']})+'\n')
        args = argparse.Namespace(group=2, temperature=.5, steps=1, lr=.01, beta=.001, clip=.2,
            data=str(root/'train.jsonl'), heldout=str(root/'test.jsonl'), out=str(root/'run'),
            model=str(root/'base'), context=128, max_new=8, adapter=None, four_bit=False,
            rank=2, alpha=4, search='wiki', cache=str(root/'cache'), turns=1,
            judge='em', judge_model='unused', seed=42, save_every=1)
        controlled = [Trajectory('train only', 'yes', [d]), Trajectory('train only', 'no', [
            Decision(p, [7, 2], 'answer', {})])]
        with patch('train.Agent.run', side_effect=controlled):
            train(args)
        status = json.loads((root/'run/status.json').read_text())
        assert status['optimizer_updates'] == 1
        assert (root/'run/adapter_final/adapter_model.safetensors').exists()
        print('PASS: generation, causal mask, gradients, frozen base, adapter reload, full training loop')


if __name__ == '__main__':
    main()
