# TinyGSM Fine-tuning Script

A Python script for fine-tuning language models using Unsloth, optimized for GTX 1050 Ti and similar hardware.

## Usage

```bash
python3 finetune.py <config_file.json>
```

### Examples

```bash
# Use the default algebra configuration
python3 finetune.py algebra.json

# Use a custom configuration
python3 finetune.py my_config.json

# Enable verbose output
python3 finetune.py algebra.json --verbose

# Show help
python3 finetune.py --help
```

## Configuration

The script uses JSON configuration files to set all training parameters. See `algebra.json` for an example configuration.

## Environment Variables

### Hugging Face Token

To enable model uploads to Hugging Face Hub, create a `.env` file with your token:

```bash
# Copy the example file
cp env.example .env

# Edit the file and add your token
nano .env
```

Add this line to your `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### Optional Dependencies

Install python-dotenv for .env file support:
```bash
pip install python-dotenv
```

For training metrics visualization:
```bash
pip install matplotlib pandas
```

## Training Metrics Visualization

The script automatically generates training plots when CSV logging is enabled:

- **Loss over training steps**: Shows model learning progress
- **Learning rate schedule**: Visualizes LR changes during training  
- **Gradient norm**: Monitors gradient stability
- **Combined progress view**: Loss and LR on same timeline

The plot is saved as `training_metrics_plot.png` in the model directory and uploaded to Hugging Face Hub along with the model.

## Features

- ✅ Command-line argument support
- ✅ JSON configuration files
- ✅ GTX 1050 Ti compatibility fixes
- ✅ LoRA fine-tuning for memory efficiency
- ✅ CSV logging for training metrics
- ✅ Training metrics visualization (PNG plots)
- ✅ Multiple model saving formats
- ✅ Hugging Face Hub integration
- ✅ Environment variable support

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Unsloth library
- Transformers library
- See `requirements.txt` for full list