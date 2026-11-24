# Validation performed for this delivery

These are software checks, not Bamboogle replication scores.

- Seven standard-library unit tests passed: normalized EM/aliases, group advantages and zero variance, JSON parsing, planner-only adapter routing, finish/invalid actions, tool cache/error behavior, data validation.
- Offline **real neural** test passed using a locally initialized tiny random Qwen2 and PEFT LoRA: actual generation, independently checked causal token log-probabilities, backpropagation, parameter change, frozen-base equality, adapter save/reload numerical equality.
- Actual `train.train()` passed one optimizer step using the tiny model and two controlled environment outcomes (reward 1 and 0). Actual policy forward/backward/AdamW/checkpoint path executed; environment trajectories were mocked for this test. This does not measure learned QA ability.
- Final `prepare.py` downloaded 125 Bamboogle records and 2 independent NQ training records successfully. Local JSONL preparation also passed.
- Live Wikipedia search for Citibank succeeded and returned public evidence.
- Python compilation and `run_all.sh` shell syntax checks passed.

Not performed: 7B weight download, 7B GPU rollout/training, full Bamboogle inference, Serper paid API calls, external LLM judge API calls, bitsandbytes/4-bit CUDA execution. No completed 7B training adapter or experimental Accuracy is included.

The verification machine has no CUDA GPU. Dependencies used: torch 2.6.0 CPU, transformers 4.51.3, peft 0.15.2, accelerate 1.6.0, datasets 3.5.0. Use the course GPU server for the actual experiment.
