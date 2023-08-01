<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

# ⚡ Lit-GPT

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

</div>

# ⚡ Lit-GPT

Hackable [implementation](lit_gpt/model.py) of state-of-the-art open-source large language models released under the **Apache 2.0 license**.

Supports popular public checkpoints such as:

- Meta AI [Llama 2](tutorials/download_llama_2.md)
- Stability AI [FreeWilly2](tutorials/download_freewilly_2.md)
- TII UAE [Falcon](tutorials/download_falcon.md)
- OpenLM Research [OpenLLaMA](tutorials/download_openllama.md)
- LMSYS [Vicuna](tutorials/download_vicuna.md) and [LongChat](tutorials/download_longchat.md)
- Together [RedPajama-INCITE](tutorials/download_redpajama_incite.md)
- EleutherAI [Pythia](tutorials/download_pythia.md)
- StabilityAI [StableLM](tutorials/download_stablelm.md)

This implementation extends on [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.



---

**🏆 NeurIPS 2023 Large Language Model Efficiency Challenge: 1 LLM + 1 GPU + 1 Day**

The Lit-GPT repository is the official starter kit for the [NeurIPS 2023 LLM Efficiency Challenge](https://llm-efficiency-challenge.github.io), which is a competition focused on finetuning an existing non-instruction tuned LLM for 24 hours on a single GPU. The competition has two tracks, one for the A100 and another for the 4090 GPUs.

If you are interested in participating, you can learn more about the NeurIPS LLM Efficiency Challenge on the official website [here](https://llm-efficiency-challenge.github.io). 

**The submission deadline is Oct 15th, 2023.**

---





## Lit-GPT design principles

This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original model.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** No strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

&nbsp;

## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Lit-GPT currently relies on flash attention from PyTorch nightly. Until PyTorch 2.1 is released you'll need to install nightly manually.
Luckily that is straightforward:

**On CUDA**

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```

**On CPU (incl Macs)**

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cpu --pre 'torch>=2.1.0dev'
```

**(Optional) install Flash Attention 2**

```bash
MAX_JOBS=4 pip install 'flash-attn>=2.0.0.post1' --no-build-isolation
```

All good, now install the dependencies:

```bash
pip install -r requirements.txt
```

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions, you need to download the model weights. **If you don't have them, check out our [guide](tutorials/download_stablelm.md).**

Run inference:

```bash
python generate/base.py --prompt "Hello, my name is"
```

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from the model](tutorials/inference.md).

You can also chat with the model interactively:

```bash
python chat/base.py
```

### Run large models on smaller consumer devices

We support 4-bit quantization (as in QLoRA), LLM.int8, and GPTQ.int4 inference by following [this guide](tutorials/quantize.md).

## Finetune the model

We provide a simple training scripts (`finetune/adapter.py`, `finetune/adapter_v2.py`, and `finetune/lora.py`) that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

2. Run the finetuning script

For example, you can either use

Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
python finetune/adapter.py
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
python finetune/adapter_v2.py
```

or LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
python finetune/lora.py
```

(Please see the [tutorials/finetune_adapter](tutorials/finetune_adapter.md) for details on the differences between the two adapter methods.)

The finetuning requires at least one GPU with ~12 GB memory (RTX 3060).

It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

### Finetuning How-To Guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with Adapters](tutorials/finetune_adapter.md)
- [Finetune with LoRA](tutorials/finetune_lora.md)

### Understanding Finetuning -- Conceptual Tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

## Pre-training

Porting from Lit-LLaMA in progress 👷

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pre-training](https://github.com/Lightning-AI/lit-gpt/labels/pre-training)
- [ ] [Fine-tuning](https://github.com/Lightning-AI/lit-gpt/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-gpt/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-gpt/labels/sparsification)

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment.

Unsure about contributing? Check out our [Contributing to Lit-LLaMA: A Hitchhiker’s Guide to the Quest for Fully Open-Source AI](https://lightning.ai/pages/community/tutorial/contributing-to-lit-llama-a-hitchhikers-guide-to-the-quest-for-fully-open-source-ai/) guide. The same guidelines apply to Lit-GPT.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.

## Complete guide for finetuning with LoRA  
  
This is to record all the commands and steps I used to finetune a llama2 model with LoRA.  
  
1. Get access to [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). We use the 7 billion chat mode, but the same steps should work for all models, except the GPU/Mem/Disk configurations.  
2. Get a compute instance of at least 100GB SSD storage, 48GB VRAM (may reduce the VRAM size if batchsize is set to be lower than 128 with 7b model in FP16 mode)  
3. Clone the repo  
```bash
git clone https://github.com/Lightning-AI/lit-gpt  
cd lit-gpt  
```  
4. Install dependencies   
```bash  
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev' torchaudio torchvision # Note the torchaudio and torchvision need to be up to date as well, otherwise there could be some C errors.  
pip install -r requirements.txt  
pip install huggingface_hub  
pip install sentencepiece  
```  
5. Download the model and convert the param names  
```bash  
python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf --token your_hf_token  
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf  
```
6. Download/Create datasets for training  
For simplicity, we use redpajama as an example. The download links can be found [here](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/tree/main). We download the `arxiv_sample.jsonl` file, but the same method should work for any other data.  
```bash  
mkdir data
mkdir data/RedPajama-Data-1T-Sample
wget https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/arxiv_sample.jsonl -O data/RedPajama-Data-1T-Sample/arxiv_sample.jsonl
```

We can also create the dataset using some json file with the following format:
```json
[
    {
        "instruction": "Arrange the given numbers in ascending order.",
        "input": "2, 4, 0, 8, 3",
        "output": "0, 2, 3, 4, 8"
    },
    ...
]
```

7. Generate the training/testing sets

In `scripts/prepare_any.py`, we may need to modify the following lines to account for different input file formats:
```python
with open(source_path, encoding="utf-8") as f:
    for row in tqdm(f):
        text = json.loads(row)["text"]
        data.append(text)
```

After modification, execute the following to generate the training/testing sets.  
```bash
python scripts/prepare_any.py \
        --source_path <path to source data downloaded/created in previous step> \
        --checkpoint_dir <path to model checkpoint, e.g. checkpoints/meta-llama/Llama-2-7b-chat-hf> \
        --destination_path <path to save the generated train/test set files>
```

8. Start finetuning  
We may want to modify the parameters defined in `finetune/lora.py` so that the process can fit to the compute instance we have.

Then, execute the following command to start the finetuning process:  
```bash
python finetune/lora.py \
        --data_dir <path to the folder containing the train/test files generated in previous step> \
        --checkpoint_dir <path to model checkpoint, e.g. checkpoints/meta-llama/Llama-2-7b-chat-hf> \
        --out_dir <path to save the logs, finetuned models> \
```