Failed reason: transformers==4.46.3 does not support Qwen3 architecture (KeyError: 'qwen3').
Fixed in v2 by upgrading to transformers==4.51.3 + switching to fp16 (works on both T4 and P100, since bf16 fails on Pascal).
