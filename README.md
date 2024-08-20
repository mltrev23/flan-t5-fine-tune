# Fine-tuning Flan-T5

This project aims to fine-tune the Flan-T5 language model using Hugging Face's Transformers library. By following these steps, you can fine-tune the model and use it for inference.

---

## Prerequisites

Before getting started, make sure you have the following:

- Hugging Face API token (HF token)
- Python installed on your system
- CUDA-enabled GPU (for training)

---

## Setup

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/mltrev23/Fine-Tuning-LLaMA-2/
    ```

2. Install the required packages using the following command:

    ```bash
    pip install -r requirements.txt
    ```

---

## Fine-tuning Flan-T5

There are four methods of fine-tuning available:

1. **Additive Fine-Tuning**: Adds layers to train.
   ```shell
   python additive-fine-tuning.py
   ```

2. **Full Fine-Tuning**: Trains all weights of the model.
   ```shell
   python full-fine-tuning.py
   ```

3. **Selective Fine-Tuning**: Chooses specific layers of the original model to train.
   ```shell
   python selective-fine-tuning.py
   ```

4. **Template Fine-Tuning**: Uses a predefined template for training. The template can be modified as needed.

---

### Custom Data Ingestion

To ingest your own data for fine-tuning, modify the code in your scripts to load your dataset. Here’s an example of loading a text dataset:

```python
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='datasets/bittensor.txt', block_size=128)
```

---

## Inference

To perform inference using the fine-tuned Flan-T5 model, you can use the following scripts:

1. **Generate Using Forward Pass**: This script demonstrates generating text using a forward pass approach.
   ```shell
   python generate-using-forward.py
   ```

2. **Main Inference Script**: This script provides a straightforward way to generate outputs from the model.
   ```shell
   python main.py
   ```

### Example Inference

You can modify the input text in the `main.py` file to test the model:

```python
input_text = ["What is FAANG?"]
```

This will generate an output based on the input provided.

---

## Happy Fine-Tuning!

Feel free to modify the scripts as needed for your specific tasks and datasets. If you encounter any issues, check the Hugging Face documentation or seek help from the community.
```

### Summary of the README Structure

- **Title and Introduction**: Describes the purpose of the project.
- **Prerequisites**: Lists necessary requirements.
- **Setup**: Instructions for cloning the repo, setting up the environment, and installing dependencies.
- **Fine-tuning Methods**: Details the four methods available for fine-tuning.
- **Custom Data Ingestion**: Provides an example of how to load custom datasets.
- **Inference**: Explains how to run inference with the fine-tuned model, including example scripts.
- **Conclusion**: Encourages users to modify scripts and seek help if needed.

Feel free to adjust any sections further based on your specific needs!