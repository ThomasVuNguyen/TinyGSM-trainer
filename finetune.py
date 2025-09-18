# GTX 1050 Ti compatibility fixes - disable compilation
import unsloth
import os
import csv
import json
import sys
import argparse
import glob
from datetime import datetime
from transformers import TrainerCallback

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune language models using Unsloth')
    parser.add_argument('config_file', 
                       help='Path to the JSON configuration file (e.g., algebra.json)')
    parser.add_argument('--verbose', '-v', 
                       action='store_true', 
                       help='Enable verbose output')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
config_file = args.config_file
verbose = args.verbose

# =============================================================================
# CONFIGURATION LOADING FROM JSON
# =============================================================================

# Load configuration from JSON file
if not os.path.exists(config_file):
    print(f"Error: Configuration file '{config_file}' not found!")
    print(f"Usage: python finetune.py <config_file.json>")
    print(f"Example: python finetune.py algebra.json")
    sys.exit(1)

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
    if verbose:
        print(f"Successfully loaded configuration from: {config_file}")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in configuration file '{config_file}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading configuration file '{config_file}': {e}")
    sys.exit(1)

# Extract configuration sections
model_config = config['model_config']
dataset_config = config['dataset_config']
lora_config = config['lora_config']
training_config = config['training_config']
inference_config = config['inference_config']
saving_config = config['saving_config']
logging_config = config['logging_config']

# Model Configuration
HUB_MODEL_NAME = model_config['hub_model_name']
MODEL_NAME = model_config['base_model_name']
MAX_SEQ_LENGTH = model_config['max_seq_length']
LOAD_IN_4BIT = model_config['load_in_4bit']
LOAD_IN_8BIT = model_config['load_in_8bit']
FULL_FINETUNING = model_config['full_finetuning']

# Dataset Configuration
DATASET_NAME = dataset_config['dataset_name']
DATASET_SPLIT = dataset_config['dataset_split']
CHAT_TEMPLATE = dataset_config['chat_template']

LORA_R = lora_config['r']
LORA_ALPHA = LORA_R * lora_config['alpha_multiplier']
LORA_DROPOUT = lora_config['dropout']
LORA_BIAS = lora_config['bias']
USE_GRADIENT_CHECKPOINTING = lora_config['use_gradient_checkpointing']
RANDOM_STATE = lora_config['random_state']
USE_RSLORA = lora_config['use_rslora']
LOFTQ_CONFIG = lora_config['loftq_config']
TARGET_MODULES = lora_config['target_modules']

# Training Configuration
PER_DEVICE_TRAIN_BATCH_SIZE = training_config['per_device_train_batch_size']
GRADIENT_ACCUMULATION_STEPS = training_config['gradient_accumulation_steps']
WARMUP_STEPS = training_config['warmup_steps']
MAX_STEPS = training_config['max_steps']
NUM_TRAIN_EPOCHS = training_config['num_train_epochs']
LEARNING_RATE = training_config['learning_rate']
WEIGHT_DECAY = training_config['weight_decay']
LR_SCHEDULER_TYPE = training_config['lr_scheduler_type']
SEED = training_config['seed']
OUTPUT_DIR = training_config['output_dir']
REPORT_TO = training_config['report_to']
OPTIM = training_config['optim']
LOGGING_STEPS = training_config['logging_steps']
SAVE_STEPS = training_config.get('save_steps', 500)  # Default to 500 if not specified

# Inference Configuration
MAX_NEW_TOKENS = inference_config['max_new_tokens']
TEMPERATURE = inference_config['temperature']
TOP_P = inference_config['top_p']
TOP_K = inference_config['top_k']
DO_SAMPLE = inference_config['do_sample']

# Model Saving Configuration
SAVE_LOCAL = saving_config['save_local']
SAVE_16BIT = saving_config['save_16bit']
SAVE_4BIT = saving_config['save_4bit']
SAVE_LORA = saving_config['save_lora']
PUSH_TO_HUB = saving_config['push_to_hub']

# CSV Logging Configuration
CSV_LOG_ENABLED = logging_config['csv_log_enabled']
CSV_LOG_FILE = f"{HUB_MODEL_NAME}/training_metrics.csv"

# Print configuration summary
print("="*60)
print("TINYGSM FINE-TUNING SCRIPT")
print("="*60)
print(f"Configuration file: {config_file}")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_NAME}")
print(f"Max Steps: {MAX_STEPS}")
print(f"Learning Rate: {LEARNING_RATE}")
print("="*60)
print("Note: To enable Hugging Face upload, create a .env file with HF_TOKEN=your_token")
print("Example: cp env.example .env && nano .env")
print("="*60)

# Available Models (for reference)
FOURBIT_MODELS = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
]  # More models at https://huggingface.co/unsloth

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    if os.path.exists('.env'):
        print("Environment variables loaded from .env file")
    else:
        print("No .env file found. Using system environment variables only.")
        print("To use .env file: cp env.example .env && nano .env")
except ImportError:
    print("python-dotenv not installed. Using system environment variables only.")
    print("To install: pip install python-dotenv")

# Get Hugging Face token from environment (.env file)
hf_token = os.getenv("HF_TOKEN")

# Set CUDA environment variables for GTX 1050 Ti compatibility
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_INDUCTOR"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
import torch

# Disable Triton and dynamic compilation
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()
torch._dynamo.config.disable = True

# =============================================================================
# CSV LOGGING CALLBACK
# =============================================================================

def generate_training_plot(csv_file_path, output_path=None):
    """Generate training metrics plot from CSV file"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        import os
        
        if not os.path.exists(csv_file_path):
            print(f"Warning: CSV file not found at {csv_file_path}")
            return None
            
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        if len(df) == 0:
            print("Warning: CSV file is empty")
            return None
            
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set up the plot with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss over steps
        if 'loss' in df.columns and 'step' in df.columns:
            ax1.plot(df['step'], df['loss'], 'b-', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)
        
        # Plot 2: Learning Rate over steps
        if 'learning_rate' in df.columns and 'step' in df.columns:
            ax2.plot(df['step'], df['learning_rate'], 'g-', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot 3: Gradient Norm over steps
        if 'grad_norm' in df.columns and 'step' in df.columns:
            ax3.plot(df['step'], df['grad_norm'], 'r-', linewidth=2, alpha=0.8)
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Gradient Norm')
            ax3.set_title('Gradient Norm')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(bottom=0)
        
        # Plot 4: Training Progress (Loss + Learning Rate on secondary y-axis)
        if all(col in df.columns for col in ['loss', 'learning_rate', 'step']):
            # Primary y-axis for loss
            color = 'tab:blue'
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss', color=color)
            line1 = ax4.plot(df['step'], df['loss'], color=color, linewidth=2, alpha=0.8, label='Loss')
            ax4.tick_params(axis='y', labelcolor=color)
            ax4.set_ylim(bottom=0)
            
            # Secondary y-axis for learning rate
            ax4_twin = ax4.twinx()
            color = 'tab:green'
            ax4_twin.set_ylabel('Learning Rate', color=color)
            line2 = ax4_twin.plot(df['step'], df['learning_rate'], color=color, linewidth=2, alpha=0.8, label='Learning Rate')
            ax4_twin.tick_params(axis='y', labelcolor=color)
            ax4_twin.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            ax4.set_title('Training Progress')
            ax4.grid(True, alpha=0.3)
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = csv_file_path.replace('.csv', '_plot.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Training plot saved to: {output_path}")
        return output_path
        
    except ImportError as e:
        print(f"Warning: Required plotting libraries not installed: {e}")
        print("To enable plotting, install: pip install matplotlib pandas")
        return None
    except Exception as e:
        print(f"Warning: Failed to generate training plot: {e}")
        return None

def upload_csv_to_hub(model_name, csv_file_path, hf_token):
    """Upload CSV file to Hugging Face Hub repository"""
    try:
        from huggingface_hub import HfApi
        import os
        
        if os.path.exists(csv_file_path):
            api = HfApi()
            api.upload_file(
                path_or_fileobj=csv_file_path,
                path_in_repo="training_metrics.csv",
                repo_id=model_name,
                token=hf_token
            )
            print(f"Training metrics CSV uploaded to {model_name}")
        else:
            print(f"Warning: CSV file not found at {csv_file_path}")
    except Exception as e:
        print(f"Warning: Failed to upload CSV file: {e}")

def upload_plot_to_hub(model_name, plot_file_path, hf_token):
    """Upload training plot PNG to Hugging Face Hub repository"""
    try:
        from huggingface_hub import HfApi
        import os
        
        if os.path.exists(plot_file_path):
            api = HfApi()
            api.upload_file(
                path_or_fileobj=plot_file_path,
                path_in_repo="training_plot.png",
                repo_id=model_name,
                token=hf_token
            )
            print(f"Training plot uploaded to {model_name}")
        else:
            print(f"Warning: Plot file not found at {plot_file_path}")
    except Exception as e:
        print(f"Warning: Failed to upload plot file: {e}")

class CSVMetricsCallback(TrainerCallback):
    """Callback to log training metrics to CSV file"""
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.metrics_data = []
        self.fieldnames = ['step', 'epoch', 'loss', 'grad_norm', 'learning_rate', 'timestamp']
        
        # Create CSV file with headers
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs is not None and CSV_LOG_ENABLED:
            # Extract metrics from logs
            metrics = {
                'step': state.global_step,
                'epoch': logs.get('epoch', 0),
                'loss': logs.get('loss', None),
                'grad_norm': logs.get('grad_norm', None),
                'learning_rate': logs.get('learning_rate', None),
                'timestamp': datetime.now().isoformat()
            }
            
            # Only log if we have meaningful data
            if any(v is not None for v in [metrics['loss'], metrics['grad_norm'], metrics['learning_rate']]):
                self.metrics_data.append(metrics)
                
                # Write to CSV file
                with open(self.csv_file_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow(metrics)
                
                print(f"Logged metrics: Step {metrics['step']}, Loss: {metrics['loss']}, LR: {metrics['learning_rate']}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        if CSV_LOG_ENABLED:
            print(f"\nTraining metrics saved to: {self.csv_file_path}")
            print(f"Total logged entries: {len(self.metrics_data)}")

# =============================================================================
# MODEL LOADING
# =============================================================================

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_8bit=LOAD_IN_8BIT,
    full_finetuning=FULL_FINETUNING,
    token=hf_token,  # Use HF token from .env file for gated models
)

"""We now add LoRA adapters so we only need to update a small amount of parameters!"""

model = FastModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    random_state=RANDOM_STATE,
    use_rslora=USE_RSLORA,
    loftq_config=LOFTQ_CONFIG,
)

"""<a name="Data"></a>
### Data Prep
We now use the `Gemma-3` format for conversation style finetunes. We use the reformatted [Tulu-3 SFT Personas Instruction Following](https://huggingface.co/datasets/ThomasTheMaker/tulu-3-sft-personas-instruction-following) dataset. Gemma-3 renders multi turn conversations like below:

```
<bos><start_of_turn>user
Hello!<end_of_turn>
<start_of_turn>model
Hey there!<end_of_turn>
```

We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template=CHAT_TEMPLATE,
)

from datasets import load_dataset
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

"""We now use `convert_to_chatml` to convert the reformatted dataset (with input/output/system columns) to the correct format for finetuning purposes!"""

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": ""},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]}
        ]
    }

dataset = dataset.map(
    convert_to_chatml
)

"""Let's see how row 100 looks like!"""

dataset[100]

"""We now have to apply the chat template for `Gemma3` onto the conversations, and save it to `text`."""

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

"""Let's see how the chat template did!

"""

dataset[100]['text']

"""<a name="Train"></a>
### Train the model
Now let's train our model. We do 100 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
"""

from trl import SFTTrainer, SFTConfig

# Create model directory and copy config
import shutil
os.makedirs(HUB_MODEL_NAME, exist_ok=True)

# Copy the JSON config to the model folder for reproducibility
config_copy_path = f"{HUB_MODEL_NAME}/config.json"
shutil.copy2(config_file, config_copy_path)
print(f"Configuration copied to: {config_copy_path}")

# Initialize CSV logging callback
csv_callback = None
if CSV_LOG_ENABLED:
    csv_callback = CSVMetricsCallback(CSV_LOG_FILE)
    print(f"CSV logging enabled. Metrics will be saved to: {CSV_LOG_FILE}")

# Prepare training arguments - handle max_steps properly
training_args = {
    "dataset_text_field": "text",
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "learning_rate": LEARNING_RATE,
    "logging_steps": LOGGING_STEPS,
    "save_steps": SAVE_STEPS,
    "optim": OPTIM,
    "weight_decay": WEIGHT_DECAY,
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "seed": SEED,
    "output_dir": OUTPUT_DIR,
    "report_to": REPORT_TO,
}

# Add either max_steps OR num_train_epochs, not both
if MAX_STEPS and MAX_STEPS > 0:
    training_args["max_steps"] = MAX_STEPS
else:
    training_args["num_train_epochs"] = NUM_TRAIN_EPOCHS

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,  # Can set up evaluation!
    args=SFTConfig(**training_args),
)

# Add CSV callback to trainer
if csv_callback:
    trainer.add_callback(csv_callback)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

"""Let's verify masking the instruction part is done! Let's print the 100th row again."""

tokenizer.decode(trainer.train_dataset[100]["input_ids"])

"""Now let's print the masked out example - you should see only the answer is present:"""

tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

# Check if there's a checkpoint to resume from
checkpoint_dirs = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
if checkpoint_dirs:
    # Sort by checkpoint number and get the latest one
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    trainer_stats = trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("No checkpoint found, starting fresh training")
    trainer_stats = trainer.train(resume_from_checkpoint=False)

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Generate training plot if CSV logging was enabled
plot_file_path = None
if CSV_LOG_ENABLED and os.path.exists(CSV_LOG_FILE):
    print("\nGenerating training metrics plot...")
    plot_file_path = generate_training_plot(CSV_LOG_FILE)

# @title Show CSV metrics summary
if CSV_LOG_ENABLED and csv_callback and csv_callback.metrics_data:
    print("\n" + "="*50)
    print("TRAINING METRICS SUMMARY")
    print("="*50)
    
    # Calculate summary statistics
    losses = [m['loss'] for m in csv_callback.metrics_data if m['loss'] is not None]
    learning_rates = [m['learning_rate'] for m in csv_callback.metrics_data if m['learning_rate'] is not None]
    grad_norms = [m['grad_norm'] for m in csv_callback.metrics_data if m['grad_norm'] is not None]
    
    if losses:
        print(f"Final Loss: {losses[-1]:.4f}")
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
        print(f"Min Loss: {min(losses):.4f}")
        print(f"Max Loss: {max(losses):.4f}")
    
    if learning_rates:
        print(f"Final Learning Rate: {learning_rates[-1]:.2e}")
        print(f"Initial Learning Rate: {learning_rates[0]:.2e}")
    
    if grad_norms:
        print(f"Final Gradient Norm: {grad_norms[-1]:.4f}")
        print(f"Average Gradient Norm: {sum(grad_norms)/len(grad_norms):.4f}")
    
    print(f"Total Logged Steps: {len(csv_callback.metrics_data)}")
    print(f"CSV File: {CSV_LOG_FILE}")
    print("="*50)

"""<a name="Inference"></a>
### Inference
Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`
"""

messages = [
    {'role': 'system','content':dataset['conversations'][10][0]['content']},
    {"role" : 'user', 'content' : dataset['conversations'][10][1]['content']}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
).removeprefix('<bos>')

from transformers import TextStreamer
# Fix cache compatibility issue by using a different generation approach
inputs = tokenizer(text, return_tensors = "pt").to("cuda")
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    do_sample=DO_SAMPLE,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=False,  # Disable cache to avoid compatibility issues
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated response:")
print(generated_text)

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

# Local saving
if SAVE_LOCAL:
    model.save_pretrained(HUB_MODEL_NAME)
    tokenizer.save_pretrained(HUB_MODEL_NAME)
    print(f"Model and tokenizer saved locally to {HUB_MODEL_NAME}")

# Use the Hugging Face token loaded earlier
if PUSH_TO_HUB and hf_token:
    model.push_to_hub(HUB_MODEL_NAME, token=hf_token)
    tokenizer.push_to_hub(HUB_MODEL_NAME, token=hf_token)
    print("Model and tokenizer uploaded to Hugging Face Hub")
    
    # Upload CSV file and plot if they exist
    if CSV_LOG_ENABLED:
        upload_csv_to_hub(HUB_MODEL_NAME, CSV_LOG_FILE, hf_token)
        if plot_file_path:
            upload_plot_to_hub(HUB_MODEL_NAME, plot_file_path, hf_token)
        
elif PUSH_TO_HUB and not hf_token:
    print("Warning: HF_TOKEN not found in .env file or environment variables.")
    print("To enable Hugging Face upload, add 'HF_TOKEN=your_token_here' to your .env file")
    print("or set the HF_TOKEN environment variable.")

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gemma-3-270m-tulu-3-sft-personas-instruction-following", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = False,
    )

"""### Saving to float16 for VLLM

We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
"""

# Merge to 16bit
if SAVE_16BIT:
    model.save_pretrained_merged(f"{HUB_MODEL_NAME}-16bit", tokenizer, save_method="merged_16bit")
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub_merged(f"{HUB_MODEL_NAME}-16bit", tokenizer, save_method="merged_16bit", token=hf_token)
        print("16-bit merged model uploaded to Hugging Face Hub")
        
        # Upload CSV file and plot if they exist
        if CSV_LOG_ENABLED:
            upload_csv_to_hub(f"{HUB_MODEL_NAME}-16bit", CSV_LOG_FILE, hf_token)
            if plot_file_path:
                upload_plot_to_hub(f"{HUB_MODEL_NAME}-16bit", plot_file_path, hf_token)
            
    elif PUSH_TO_HUB and not hf_token:
        print("Warning: HF_TOKEN not found in .env file. Skipping 16-bit model upload to Hugging Face.")

# Merge to 4bit
if SAVE_4BIT:
    model.save_pretrained_merged(f"{HUB_MODEL_NAME}-4bit", tokenizer, save_method="merged_4bit")
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub_merged(f"{HUB_MODEL_NAME}-4bit", tokenizer, save_method="merged_4bit", token=hf_token)
        print("4-bit merged model uploaded to Hugging Face Hub")
        
        # Upload CSV file and plot if they exist
        if CSV_LOG_ENABLED:
            upload_csv_to_hub(f"{HUB_MODEL_NAME}-4bit", CSV_LOG_FILE, hf_token)
            if plot_file_path:
                upload_plot_to_hub(f"{HUB_MODEL_NAME}-4bit", plot_file_path, hf_token)
            
    elif PUSH_TO_HUB and not hf_token:
        print("Warning: HF_TOKEN not found in .env file. Skipping 4-bit model upload to Hugging Face.")

# Just LoRA adapters
if SAVE_LORA:
    model.save_pretrained(f"{HUB_MODEL_NAME}-lora")
    tokenizer.save_pretrained(f"{HUB_MODEL_NAME}-lora")
    if PUSH_TO_HUB and hf_token:
        model.push_to_hub(f"{HUB_MODEL_NAME}-lora", token=hf_token)
        tokenizer.push_to_hub(f"{HUB_MODEL_NAME}-lora", token=hf_token)
        print("LoRA adapters uploaded to Hugging Face Hub")
        
        # Upload CSV file and plot if they exist
        if CSV_LOG_ENABLED:
            upload_csv_to_hub(f"{HUB_MODEL_NAME}-lora", CSV_LOG_FILE, hf_token)
            if plot_file_path:
                upload_plot_to_hub(f"{HUB_MODEL_NAME}-lora", plot_file_path, hf_token)

print("\n" + "="*60)
print("FINE-TUNING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Configuration used: {config_file}")
print(f"Model saved to: {HUB_MODEL_NAME}")
print("="*60)