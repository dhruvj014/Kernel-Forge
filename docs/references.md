# References

Papers and repositories that informed the design. Code was written from scratch. Repositories listed here were read for ideas only, and any that appear in benchmarks are used strictly as baselines.

## Papers

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* 2022. https://arxiv.org/abs/2205.14135
- Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. https://arxiv.org/abs/2307.08691
- Maxim Milakov, Natalia Gimelshein. *Online normalizer calculation for softmax.* 2018. https://arxiv.org/abs/1805.02867
- Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* 2022. https://arxiv.org/abs/2210.17323
- Ji Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* 2023. https://arxiv.org/abs/2306.00978
- Samuel Williams, Andrew Waterman, David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM, 2009.

## NVIDIA documentation

- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA C++ Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- PTX ISA (mma.sync, cp.async, ldmatrix): https://docs.nvidia.com/cuda/parallel-thread-execution/
- Nsight Compute documentation: https://docs.nvidia.com/nsight-compute/

## Repositories (ideas or baselines only, no code copied)

- FlashAttention: https://github.com/Dao-AILab/flash-attention (optional benchmark baseline)
- GPTQ: https://github.com/IST-DASLab/gptq
- AWQ: https://github.com/mit-han-lab/llm-awq
- Marlin: https://github.com/IST-DASLab/marlin (read for INT4 kernel ideas)

## Models and data

- TinyLlama 1.1B Chat: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Qwen2.5 1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
- WikiText-2 (raw): https://huggingface.co/datasets/Salesforce/wikitext

This list grows phase by phase as each source is actually used.
