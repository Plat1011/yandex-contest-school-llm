Training completed (~3h on T4), loss 0.35 → 0.29. Then crashed on vLLM import:
ImportError: vllm/_C.abi3.so: undefined symbol _ZN3c104cuda29c10_cuda_check_implementation...

ABI mismatch between vllm==0.11.0 wheel and Kaggle's pre-installed torch.

merged_weights are saved on Kaggle but full salvage requires uploading weights as Kaggle Dataset which is too slow on user connection.

Next: v3 retrains from scratch using transformers.generate for eval (no vllm at all in training kernel).
