# VRAM Optimization Guide for TinyGSM Training

This guide provides recommendations for modifying `og-gsm.json` based on your available VRAM. The current configuration is optimized for 16GB+ VRAM.

## Quick Reference by VRAM Amount

### 8GB-12GB VRAM (Low VRAM)
- **Model Loading**: Use 4-bit quantization
- **Training**: LoRA-only, small batch sizes
- **Sequence Length**: 512-1024 tokens
- **Batch Size**: 1-4 per device

### 12GB-16GB VRAM (Medium VRAM)
- **Model Loading**: Use 8-bit quantization
- **Training**: LoRA with some full finetuning
- **Sequence Length**: 1024-2048 tokens
- **Batch Size**: 4-8 per device

### 16GB-24GB VRAM (High VRAM)
- **Model Loading**: Current settings work well
- **Training**: Full finetuning or LoRA
- **Sequence Length**: 2048 tokens (current)
- **Batch Size**: 8-16 per device

### 24GB+ VRAM (Very High VRAM)
- **Model Loading**: Full precision
- **Training**: Full finetuning
- **Sequence Length**: 4096+ tokens
- **Batch Size**: 16+ per device

## Detailed Parameter Modifications

### Model Configuration (`model_config`)

| Parameter | 8GB-12GB | 12GB-16GB | 16GB-24GB | 24GB+ |
|-----------|-----------|-----------|-----------|-------|
| `max_seq_length` | 512-1024 | 1024-2048 | 2048 | 4096+ |
| `load_in_4bit` | `true` | `false` | `false` | `false` |
| `load_in_8bit` | `false` | `true` | `false` | `false` |
| `full_finetuning` | `false` | `false` | `true` | `true` |

### LoRA Configuration (`lora_config`)

| Parameter | 8GB-12GB | 12GB-16GB | 16GB-24GB | 24GB+ |
|-----------|-----------|-----------|-----------|-------|
| `r` | 64-128 | 128-256 | 256 | 512+ |
| `alpha_multiplier` | 2 | 2 | 2 | 2 |
| `dropout` | 0.2-0.3 | 0.1-0.2 | 0.1 | 0.05-0.1 |
| `use_gradient_checkpointing` | `true` | `true` | `false` | `false` |
| `loftq_config` | `{"loftq_bits": 4}` | `null` | `null` | `null` |
| `target_modules` | `["q_proj", "v_proj"]` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | All modules | All modules |

### Training Configuration (`training_config`)

| Parameter | 8GB-12GB | 12GB-16GB | 16GB-24GB | 24GB+ |
|-----------|-----------|-----------|-----------|-------|
| `per_device_train_batch_size` | 1-4 | 4-8 | 8-16 | 16+ |
| `gradient_accumulation_steps` | 8-16 | 4-8 | 2-4 | 1-2 |
| `warmup_steps` | 50-100 | 25-50 | 10-25 | 5-10 |
| `num_train_epochs` | 0.5-0.8 | 0.8-1.0 | 1.0 | 1.0-3.0 |
| `learning_rate` | 1e-4 | 5e-5 | 5e-5 | 1e-5 |
| `weight_decay` | 0.1 | 0.05 | 0.01 | 0.001-0.01 |
| `optim` | `"adamw_8bit"` | `"adamw_8bit"` | `"adamw_8bit"` | `"adamw"` |
| `save_steps` | 500-1000 | 1000-2000 | 2000 | 2000-5000 |

### Inference Configuration (`inference_config`)

| Parameter | 8GB-12GB | 12GB-16GB | 16GB-24GB | 24GB+ |
|-----------|-----------|-----------|-----------|-------|
| `max_new_tokens` | 50-75 | 75-125 | 125 | 200+ |
| `temperature` | 0.7-0.8 | 0.8-1.0 | 1.0 | 1.0 |
| `top_p` | 0.8-0.9 | 0.9-0.95 | 0.95 | 0.95 |
| `top_k` | 32-40 | 40-64 | 64 | 64+ |
| `do_sample` | `false` | `true` | `true` | `true` |

### Saving Configuration (`saving_config`)

| Parameter | 8GB-12GB | 12GB-16GB | 16GB-24GB | 24GB+ |
|-----------|-----------|-----------|-----------|-------|
| `save_16bit` | `true` | `false` | `false` | `false` |
| `save_4bit` | `true` | `false` | `false` | `false` |
| `save_lora` | `true` | `true` | `false` | `false` |
| `push_to_hub` | `false` | `true` | `true` | `true` |

## Memory Usage Estimation

### Approximate VRAM Usage by Configuration:

- **8GB-12GB setup**: ~6-10GB VRAM
- **12GB-16GB setup**: ~10-14GB VRAM  
- **16GB-24GB setup**: ~14-20GB VRAM
- **24GB+ setup**: ~20-30GB VRAM

*Note: Actual usage may vary based on model size, sequence length, and other factors.*

## Tips for Low VRAM Systems

1. **Start with LoRA-only training** - Much more memory efficient
2. **Use gradient checkpointing** - Trades compute for memory
3. **Reduce sequence length** - Quadratic memory scaling with sequence length
4. **Use smaller batch sizes** - Linear memory scaling with batch size
5. **Enable quantization** - 4-bit or 8-bit quantization saves significant memory
6. **Save more frequently** - Prevents losing progress if OOM occurs

## Tips for High VRAM Systems

1. **Increase batch size** - Better gradient estimates and faster training
2. **Use longer sequences** - Better for complex reasoning tasks
3. **Full finetuning** - Better performance than LoRA for large models
4. **Higher learning rates** - Can afford more aggressive training
5. **Multiple epochs** - Can train longer for better convergence

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length`
4. Enable `load_in_4bit` or `load_in_8bit`
5. Enable `use_gradient_checkpointing`

### Slow Training
1. Increase `per_device_train_batch_size` if you have memory headroom
2. Reduce `gradient_accumulation_steps`
3. Disable `use_gradient_checkpointing` if you have enough memory
4. Use `adamw` instead of `adamw_8bit` if you have enough memory

### Poor Model Performance
1. Increase `r` in LoRA config for more parameters
2. Add more `target_modules` in LoRA config
3. Increase `num_train_epochs`
4. Adjust `learning_rate` (try 1e-4 to 1e-5 range)
5. Increase `max_seq_length` for better context understanding
