"""Batching, log-probs, advantages and losses (PPO + GRPO) with retrieved-token masking.

Batch layout (same as veRL):   [ left-padded prompt (P) | right-padded response (R) ]
so the response always occupies the last R columns. `resp_mask` marks policy-
generated tokens (1) vs. retrieved / environment tokens and padding (0).
"""
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- batching
def collate(trajs, pad_id: int, device):
    P = max(len(t.prompt_ids) for t in trajs)
    R = max(max(len(t.resp_ids) for t in trajs), 1)
    ids, am, rmask, rattn = [], [], [], []
    for t in trajs:
        p, r = t.prompt_ids, t.resp_ids
        ids.append([pad_id] * (P - len(p)) + p + r + [pad_id] * (R - len(r)))
        am.append([0] * (P - len(p)) + [1] * (len(p) + len(r)) + [0] * (R - len(r)))
        rmask.append(t.resp_mask + [0] * (R - len(r)))
        rattn.append([1] * len(r) + [0] * (R - len(r)))
    am_t = torch.tensor(am, device=device)
    return {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": am_t,
        "position_ids": (am_t.cumsum(-1) - 1).clamp(min=0),
        "resp_mask": torch.tensor(rmask, device=device, dtype=torch.float32),
        "resp_attn": torch.tensor(rattn, device=device),
        "prompt_len": P,
        "resp_len": R,
    }


def index_batch(batch, idx):
    return {k: (v[idx] if torch.is_tensor(v) else v) for k, v in batch.items()}


def _trim(batch, sl):
    """Drop all-padding columns of a micro-batch; return inputs and its response length r."""
    am = batch["attention_mask"][sl]
    start = int((am.sum(0) > 0).nonzero()[0])
    r = max(int(batch["resp_attn"][sl].sum(-1).max()), 1)
    end = batch["prompt_len"] + r
    sub = {k: batch[k][sl][:, start:end] for k in ("input_ids", "attention_mask", "position_ids")}
    return sub, r


def _pad_to(x, R):
    return F.pad(x, (0, R - x.shape[1]))


def micro_slices(n, mb):
    return [slice(i, min(i + mb, n)) for i in range(0, n, mb)]


def response_logprobs(model, batch, sl):
    """log pi(y_t | x, y_<t) for every response position of the micro-batch `sl` -> (mb, R)."""
    sub, r = _trim(batch, sl)
    out = model(**sub, use_cache=False, logits_to_keep=r + 1)
    logits = out.logits[:, :-1].float()
    tgt = sub["input_ids"][:, -r:]
    logp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return _pad_to(logp, batch["resp_len"])


def critic_values(critic, batch, sl):
    sub, r = _trim(batch, sl)
    return _pad_to(critic(**sub, resp_len=r), batch["resp_len"])


@torch.no_grad()
def batched(fn, model, batch, mb):
    n = batch["input_ids"].shape[0]
    return torch.cat([fn(model, batch, sl) for sl in micro_slices(n, mb)], 0)


# ----------------------------------------------------------------------------- reductions
def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x, mask, eps=1e-8):
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask


def seq_mean_token_mean(x, mask):
    """1/|y| sum over policy tokens, then average over sequences (paper Eq. 2-3)."""
    tok = mask.sum(-1)
    per_seq = (x * mask).sum(-1) / tok.clamp(min=1)
    valid = (tok > 0).float()
    return (per_seq * valid).sum() / valid.sum().clamp(min=1)


# ----------------------------------------------------------------------------- advantages
def last_token_index(mask):
    pos = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return (pos * (mask > 0)).argmax(-1)


def outcome_to_token_rewards(scores, mask):
    """Put the scalar outcome reward on the last policy-generated token."""
    r = torch.zeros_like(mask)
    has = mask.sum(-1) > 0
    idx = last_token_index(mask)
    r[has, idx[has]] = scores[has]
    return r


def gae_advantages(token_rewards, values, mask, gamma=1.0, lam=1.0):
    """GAE computed only over policy tokens: retrieved tokens are skipped, i.e. they are
    treated as part of the environment transition between two policy tokens."""
    B, R = mask.shape
    adv = torch.zeros_like(values)
    last_gae = torch.zeros(B, device=values.device)
    next_v = torch.zeros(B, device=values.device)
    for t in reversed(range(R)):
        m = mask[:, t]
        delta = token_rewards[:, t] + gamma * next_v - values[:, t]
        gae = delta + gamma * lam * last_gae
        adv[:, t] = gae * m
        last_gae = m * gae + (1 - m) * last_gae
        next_v = m * values[:, t] + (1 - m) * next_v
    returns = (adv + values) * mask
    return masked_whiten(adv, mask), returns


def grpo_advantages(scores, group_ids, mask, eps=1e-6):
    """(r_i - mean(group)) / (std(group) + eps), broadcast to every policy token."""
    adv = torch.zeros_like(scores)
    for g in torch.unique(group_ids):
        sel = group_ids == g
        s = scores[sel]
        std = s.std() if s.numel() > 1 else torch.tensor(1.0, device=s.device)
        adv[sel] = (s - s.mean()) / (std + eps)
    return adv.unsqueeze(-1) * mask


# ----------------------------------------------------------------------------- losses
def kl_estimate(logp, ref_logp, kind):
    if kind == "k1":                       # used inside the PPO reward
        return logp - ref_logp
    if kind == "low_var_kl":               # k3, used as GRPO loss term
        d = ref_logp - logp
        return (torch.exp(d) - d - 1).clamp(-10, 10)
    raise ValueError(kind)


def ppo_policy_loss(logp, old_logp, adv, mask, clip=0.2):
    ratio = torch.exp(logp - old_logp)
    loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip))
    clipfrac = masked_mean(((ratio - 1).abs() > clip).float(), mask)
    return seq_mean_token_mean(loss, mask), clipfrac


def value_loss(values, old_values, returns, mask, clip=0.5):
    v_clip = old_values + (values - old_values).clamp(-clip, clip)
    loss = torch.max((values - returns) ** 2, (v_clip - returns) ** 2)
    return 0.5 * masked_mean(loss, mask)
