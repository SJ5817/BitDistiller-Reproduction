# BitDistiller Reproduction and Modification for MetaMath-7B-V1.0

This repository presents a reproduction of the original [BitDistiller](https://github.com/DD-DuDa/BitDistiller) work. It includes modifications to the codebase to enable 2-bit quantization and gsm8k testing specifically for the [MetaMath-7B-V1.0](https://huggingface.co/meta-math/MetaMath-7B-V1.0) model.

---

## Related Work: BitDistiller Reproduction for Llama3.2-1B (by Teammate)

My teammate, [EricBennetts](https://github.com/EricBennetts), has also successfully reproduced the BitDistiller methodology, focusing on the Llama3.2-1B model. His work includes:

*   2-bit quantization of Llama3.2-1B.
*   Perplexity (PPL) evaluation of the quantized model.
*   Inference speed comparison.

You can find their complete implementation and results here:
➡️ **[BitDistiller-Reproduced by EricBennetts](https://github.com/EricBennetts/BitDistiller-Reproduced)**

---

## Project Setup: MetaMath-7B-V1.0 Reproduction

This section details the environment and hardware used for the MetaMath-7B-V1.0 reproduction.

### Hardware Configuration

*   **GPU:** 4x NVIDIA A40 (48GB each)
*   **CPU:** 60-core AMD EPYC 7543 32-Core Processor
*   **RAM:** 320GB

### Software Environment

*   **Python:** `3.9.x`
*   **PyTorch:** Approximately `2.5.1` (with CUDA 12.1 support).
    *   *Note: The exact PyTorch minor version is not precisely recalled. It is crucial to install a PyTorch version compatible with **CUDA 12.1**. You can find the correct installation command on the [official PyTorch website](https://pytorch.org/get-started/locally/) by selecting your OS, package manager (pip/conda), Python 3.9, and CUDA 12.1. For example, a compatible version might look like `torch==2.2.0+cu121` or similar.*
*   **CUDA Toolkit:** `12.1` (Ensure your NVIDIA drivers are also compatible)

**⚠️ Important Note for `transformers` library:**

If you intend to use a `transformers` library version **greater than `4.46.0`** (e.g., `4.46.1` or newer), you **must** modify the `./train/train.sh` script.

Specifically, you will need to change the argument name related to evaluation steps in ./train/train.sh:
*   **Change 'evaluation_step' to 'eval_step'**, as 'evaluation_step' is deprecated in `transformers` v4.46.0 and later.

## Key Results for MetaMath-7B-V1.0 (2-bit Quantization)

After 2-bit quantization (g128 configuration), an accuracy of approximately **44.44%** was achieved on the gsm8k test set (583 correct out of 1312 questions). This result is comparable to the accuracy reported in the original BitDistiller paper (51.02% for their model and setup).

| **MetaMath-7B-V1.0**     | **GSM8K**                 |
| :----------------------: | :-----------------------: |
|                          | Accuracy (correct/error)  |
| **2bit/g128**            | 44.43% (583/1312)         |

## Training Details for MetaMath-7B-V1.0

*   The model was trained for **2 epochs**.
*   The total training time was approximately **2-3 hours**.
*   The original DeepSpeed parameters were largely unchanged, with the exception of:
    *   `--per_device_train_batch_size 4`
    *   `--per_device_eval_batch_size 4`
    *   `--gradient_accumulation_steps 4`
*   All 4 GPUs utilized approximately **40GB of VRAM** each.
*   System RAM usage was around **160GB** (due to optimizer offloading - DeepSpeed Stage 2).
*   To manage storage space (as each checkpoint is around 100GB) and save training time, the following Hugging Face Trainer arguments were adjusted:
    *   `--eval_steps 10`
    *   `--save_strategy "steps"`
    *   `--save_steps 10`
    *   `--save_total_limit 2`
*   The training loss remained around **20** in the later stages.

For more detailed training configurations and scripts, please refer to the `train/train.sh` file.

## Code Modifications & Configuration Notes for MetaMath-7B-V1.0

The following adjustments and configurations are crucial for running this specific reproduction.

### 1. GPU Configuration in `test.py`

*   **File:** `./test/gsm8k/test.py`
*   **Action:** Users need to adjust the GPU count and `max_memory` settings (around line 202) to match their hardware.
    ```python
    # Original relevant snippet in test.py (around line 202)
    n_gpus = torch.cuda.device_count()
    max_memory = f'80000MB' # Needs adjustment based on your GPU VRAM
    max_memory = {i: max_memory for i in range(n_gpus)}
    ```

### 2. GPU Count in `train.sh`

*   **File:** `./train/train.sh`
*   **Action:** Ensure the GPU count specified or implied in `train.sh` (e.g., via `CUDA_VISIBLE_DEVICES` or DeepSpeed arguments like `--num_gpus`) matches your available hardware.

### 3. Checkpoint Resumption Implementation

The original `train.py` lacked explicit logic for resuming from saved checkpoints. The following modifications were made in `./train/train.py`:

*   **Added `find_latest_checkpoint` function:**
    ```python
    import glob # Ensure glob is imported
    import os   # Ensure os is imported

    def find_latest_checkpoint(output_dir):
        """Find the latest checkpoint in the output directory."""
        if not os.path.exists(output_dir):
            return None
        
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        
        # Sort by checkpoint number (extracting the integer part)
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoints[-1]
        
        return latest_checkpoint
    ```
*   **Modified `trainer.train()` call (around original line 384):**
    ```python
    # --- Modified call in train.py ---
    # IMPORTANT: Adjust the path to your checkpoint directory as needed.
    # This path should match the 'output_dir' in your TrainingArguments.
    checkpoint_dir = "./ckpts/MetaMath-7b/int2-g128/" # Example path, **MUST BE ADJUSTED BY USER**
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"Found and resuming from latest checkpoint: {latest_checkpoint}")
    else:
        print(f"No checkpoint found in {checkpoint_dir}. Starting training from scratch.")
    
    # Start training
    if latest_checkpoint:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()
    ```

### 4. Adjustment in `gsm8k/eval.py`

*   **File:** `./test/gsm8k/eval.py`
*   **Issue:** A potential issue was identified in the `eval_json` function where the variable `origin_json_path` could be used before assignment under certain path conditions.
*   **Action:** The relevant code block was adjusted to ensure `origin_json_path` is always initialized before use. (Users should verify this change if encountering issues or refer to the commit history for the exact modification).

### 5. Teacher Model Loading in `train.py`

*   **File:** `./train/train.py` (around line 332)
*   **Issue:** To prevent potential errors during teacher model loading with specific quantization flags.
*   **Modification:**
    Original code:
    ```python
    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
    )
    ```
    Modified code (removed explicit `load_in_4bit` and `load_in_8bit` which might conflict if `device_map` is used for distribution, and `max_memory` if not carefully set):
    ```python
    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map
        # If max_memory is still needed, ensure it's configured correctly for your setup.
    )
    ```

## Important Notice

The final configurations detailed in this "reproduction project" are **specifically tailored** for training a 2-bit quantized MetaMath-7B-V1.0 model on a 4-GPU (NVIDIA A40) setup. Adjustments will likely be necessary for different hardware or model configurations.

---

## Original BitDistiller Project

For the original BitDistiller paper, codebase, and procedures, please refer to their repository:
[DD-DuDa/BitDistiller](https://github.com/DD-DuDa/BitDistiller)

# [ACL 2024] BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation [[paper]](http://arxiv.org/abs/2402.10631)

**Implementing efficient sub-4-bit weight quantization (3 / 2 bits) in LLMs through advanced QAT-based Self-Distillation techniques.**

![overview](./imgs/overview.jpg)

## Comparing general language tasks with other methods
![overview](./imgs/result7b.jpg)
<!-- ![overview](./imgs/result2.png)-->
## Comparing reasoning benchmarks with other methods
![overview](./imgs/result.png)

## Example on 2-bit inference of a Domain-specific LLM (MetaMath)
![gif](./imgs/Bitdistiller.gif)

## News
* [2024/05] 🔥 BitDistiller has been accepted to ACL main 2024! 


## Contents
1. [Setup](#1-setup)
2. [Running](#2-running)
3. [Evaluation](#3-evaluation)
4. [Inferencce](#4-inference)

## 1. Setup
* python 3.9, pytorch >= 1.13
* pip install -r requirement.txt 
  
  (You may need to change the version of transformers according to the model config)

## 2. Running

Our results is running by following 3 steps:

### 2.1. Asymmetric Quantization
* Determine the type of quantization: use `nf3` for 3 bits and `int` for 2 bits. Set `w_bit` and `quant_type` accordingly.

* Perform clipping before training and save the clipping values using dump_clip (see `quantization/autoclip.py`).

>This step can match or surpass the low-bit PTQ quantization results of GPTQ and AWQ.

### 2.2. Generating Teacher Data
* For QAT, create data using the Teacher Model (BF16). The data varies depending on the model (see `data/generation`).


### 2.3. KD-base QAT
* Detailed procedure available in `train/`


### Example Srcipts

<details>
  <summary>LLaMA-2</summary>
  
1. Get the Clipping result
    ```bash
    cd BitDistiller/quantization

    CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path <model_path> --calib_dataset pile --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/hf-llama2-7b/int2-g128.pt
    ```
2. Get the Teacher Generation Data (Using vllm would be much faster)
    ```bash
    # vllm
    python generate_vllm.py --base_model <model_path> --dataset_name wikitext --out_path ./datasets/hf-llama-2-7b/ --max_sample 3000

    python generate_vllm.py --base_model <model_path> --dataset_name alpaca --out_path ./datasets/hf-llama-2-7b/ --max_sample 5000

    # change to path in .py
    python mix_data.py
    ```

    ```bash
    # torchrun
    cd BitDistiller/data/generation

    bash generate.sh <model_path> wikitext ../datasets/hf-llama-2-7b/ 16 3000

    bash generate.sh <model_path> alpaca ../datasets/hf-llama-2-7b/ 16 5000

    # change to path in .py
    python mix_data.py
    ```
3. Run KD-base QAT
    ```bash
    # Specify the pre-trained model path
    # Specify the num_gpus and batch_size according to your GPU devices
    # Specify the clipping cache path to the --clip

    cd train
    
    bash train.sh ../data/datasets/hf-llama-2-7b/mix_wiki_alpaca_8000.json ./ckpts/hf-llama-2-7b/int2-g128/ ./logs/hf-llama-2-7b/int2-g128/ 4
    ```
</details>

<details>
  <summary>WizardCoder</summary>
  
1. Get the Clipping result
    ```bash
    cd BitDistiller/quantization

    CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path <model_path> --calib_dataset code --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/WizardCoder-7B/int2-g128.pt
    ```
2. Get the Teacher Generation Data
    ```bash
    # vllm
    python generate_vllm.py --base_model <model_path> --dataset_name code --out_path ./datasets/WizardCoder-7b/ --max_sample 3000
    ```

    ```bash
    cd BitDistiller/data/generation

    bash generate.sh /root/WizardCoder-Python-7B/ code ../datasets/WizardCoder-7b/ 16 3000
    ```
3. Run KD-base QAT
    ```bash
    # Specify the pre-trained model path
    # Specify the num_gpus and batch_size according to your GPU devices
    # Specify the clipping cache path to the --clip

    cd train
    
    bash train.sh ../data/datasets/WizardCoder-7b/code_T0.7_N1024_S42_3000.json ./ckpts/WizardCoder-7b/int2-g128/ ./logs/WizardCoder-7b/int2-g128/ 2
    ```
</details>

<details>
  <summary>MetaMath</summary>

1. Get the Clipping result
    ```bash
    cd BitDistiller/quantization

    CUDA_VISIBLE_DEVICES=0 python autoclip.py --model_path <model_path> --calib_dataset gsm8k --quant_type int --w_bit 2 --q_group_size 128 --run_clip --dump_clip ./clip_cache/MetaMath-7B/int2-g128.pt
    ```
2. Get the Teacher Generation Data
    ```bash
    # vllm
    python generate_vllm.py --base_model <model_path> --dataset_name math --out_path ./datasets/MetaMath-7B/ --max_sample 3000
    ```

    ```bash
    cd BitDistiller/data/generation

    bash generate.sh /root/MetaMath-7B-V1.0/ math ../datasets/MetaMath-7B/ 16 3000
    ```
3. Run KD-base QAT
    ```bash
    # Specify the pre-trained model path
    # Specify the num_gpus and batch_size according to your GPU devices
    # Specify the clipping cache path to the --clip

    cd train
    
    bash train.sh ../data/datasets/MetaMath-7B/math_T0.7_N1024_S42_3000.json ./ckpts/MetaMath-7b/int2-g128/ ./logs/MetaMath-7b/int2-g128/ 2
    ```
</details>

## 3. Evaluation
### Example Srcipts
<details>
  <summary>LLaMA-2</summary>



* Test PPL on WikiText-2
  ```bash
  cd test/general

  python wiki_ppl.py --model ../../train/ckpts/hf-llama-2-7b/int2-g128/checkpoint-200/ --quant_type int --bits 2 --group_size 128
  ```
* Test MMLU
  ```bash
  CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model ../../train/ckpts/hf-llama-2-7b/int2-g128/checkpoint-200/ --eval_tasks hendrycksTest-* --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 5
  ```
* Test Common-sense QA Tasks
  ```bash
  CUDA_VISIBLE_DEVICES=0 python llm_eval.py --model ../../train/ckpts/hf-llama-2-7b/int2-g128/checkpoint-200/ --eval_tasks arc_challenge,winogrande,hellaswag,piqa --test_set --bits 2 --group_size 128 --quant_type int --num_fewshot 0 
  ```

</details>

<details>
  <summary>WizardCoder</summary>

* Install the environment according to the instructions of [HumanEval](https://github.com/openai/human-eval), 

* Example script:
    ```bash
    cd test/humaneval
    bash gen_preds.sh [checkpoint_path] ./preds/7b/int2-g128/
    ```
</details>

<details>
  <summary>MetaMath</summary>
  
* Example script:

    ```bash
    cd test/gsm8k
    bash test.sh ../../train/ckpts/MetaMath-7b/int2-g128/ ./preds/7b/int2-g128/
    ```
</details>


## 4. Inference
Please see `inference/`



## Reference
If you find BitDistiller useful or relevant to your research, please kindly cite our paper:
```
@misc{du2024bitdistiller,
      title={BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation}, 
      author={Dayou Du and Yijia Zhang and Shijie Cao and Jiaqi Guo and Ting Cao and Xiaowen Chu and Ningyi Xu},
      year={2024},
      eprint={2402.10631},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
