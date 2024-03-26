<div align="center">

<img src="assets/figures/sphinx-v_text.png" style="width: 35%" alt="SPHINX-V Logo"/>

</div>

<div align="center">

## 🎨 Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want

[Weifeng Lin](), [Xinyu Wei](), [Ruichuan An](), [Peng Gao](), [Bocheng Zou](), [Yulin Luo](), [Siyuan Huang](), [Shanghang Zhang]() and [Hongsheng Li]()

</div>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://dreamllm.github.io/) [![arXiv Paper](https://img.shields.io/badge/arxiv-2312.10032-ECA8A7?logo=arxiv)](https://arxiv.org/pdf/2312.10032.pdf) [![Static Badge](https://img.shields.io/badge/Demo-6B88E3?logo=youtubegaming&logoColor=DAE4EE)](http://111.0.123.204:8000/) [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-blue.svg)](https://github.com/RunpeiDong/ChatDreamer-Private/blob/master/LICENSE)

[[🌐 Project Page](https://mathverse-cuhk.github.io/)] [[📖 Paper](https://arxiv.org/pdf/2403.14624.pdf)] [[🤗 MDVP-Data](https://huggingface.co/datasets/AI4Math/MathVerse)] [[🤗 MDVP-Bench](https://huggingface.co/datasets/AI4Math/MathVerse)] [[🤖️ Model](https://huggingface.co/datasets/AI4Math/MathVerse)] [[🎮 Demo](https://mathverse-cuhk.github.io/#leaderboard)]

</div>

## 💥 News

- **[2024.03.28]** 🔥 We released the [SPHINX-V 13B model](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main) and [online demo](http://111.0.123.204:8000/).

- **[2024.03.28]** 🚀 We release the [arXiv paper](https://arxiv.org/pdf/2403.14624) and make a home page in the [Project Page](https://mathverse-cuhk.github.io/).

- **[2024.03.28]** 🔥 We released the [MDVP-Data](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K) dataset and [MDVP-Bench]() benchmark.

- **[2024.03.28]** 🚀 We released the traning and [evaluation](./accessory/eval/README.md) code.


## 💪 ToDo

- &nbsp;&nbsp;✅ The *testmini* set of MathVerse will be released at [🤗 Huggingface](https://huggingface.co/datasets/AI4Math/MathVerse) in a week.

- - [x] Coming soon: *CoT Evaluation results*, evaluation tools, and the entire MathVerse dataset


## 👀 Introduction

The interaction between humans and artificial intelligence (AI) is a crucial factor that reflects the effectiveness of multimodal large language models (MLLMs). However, current MLLMs primarily focus on image-level comprehension and limit interaction to textual instructions, thereby constraining their flexibility in usage and depth of response. Therefore, we introduce the **Draw-and-Understand project**: a new model, a multi-domain dataset, and a challenging benchmark for visual prompting.

<p align="center">
    <img src="assets/figures/fig1.jpg" width="90%"> <br>
</p>

Specifically, the model is named **SPHINX-V**, a new multimodal large language model designed for visual prompting, equipped with a novel visual prompt encoder and a two-stage training strategy. SPHINX-V supports multiple visual prompts simultaneously across various types, significantly enhancing user flexibility and achieve a fine-grained and open-world understanding of visual prompts.

<p align="center">
    <img src="assets/figures/fig2.jpg" width="90%"> <br>
    Six different versions of each problem in <b>MathVerse</b> transformed by expert annotators.
</p>


## 🚀 Examples Show

<details>
<summary>🔍 Natural Image Domain</summary>

<p align="center">
    <img src="assets/figures/ver1.jpg" width="100%"> <br>
</p>
</details>

<details>
<summary>🔍 OCR Image Domain</summary>

<p align="center">
    <img src="assets/figures/ver2.jpg" width="100%"> <br>
</p>
</details>

<details>
<summary>🔍 Mobile/Website Screenshot Domain</summary>

<p align="center">
    <img src="assets/figures/ver3.jpg" width="100%"> <br>
</p>
</details>

<details>
<summary>🔍 Multi-panel Image Domain</summary>

<p align="center">
    <img src="assets/figures/ver4.jpg" width="100%"> <br>
</p>
</details>


## 🛠️ Install 

1. Clone this repository and navigate to Draw-and-Understand folder
``` bash
git clone https://github.com/AFeng-x/Draw-and-Understand.git
cd Draw-and-Understand
```
2. Install packages
``` bash
# Create a new conda environment named 'sphinx-v' with Python 3.10
conda create -n sphinx-v python=3.10 -y
# Activate the 'sphinx-v' environment
conda activate sphinx-v
# Install required packages from 'requirements.txt'
pip install -r requirements.txt
```
3. Optional: Install Flash-Attention
``` bash
# Draw-and-Understand is powered by flash-attention for efficient attention computation.
pip install flash-attn --no-build-isolation
```
4. Install Draw-and-Understand as Python Packege
``` bash
# go to the root directory of Draw-and-Understand
cd Draw-and-Understand
# install Draw-and-Understand
pip install -e .
# After this, you will be able to invoke “import SPHINX_V” without the restriction of working directory.
```
5. To enable the segmentation ability shown in our official demo, SAM is also needed:
``` bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```


## 🤖️ Checkpoints

SPHINX-V-13b Stage-1 Pre-training Weight: 🤗[Hugging Face](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main) / [Baidu]()

SPHINX-V-13b Stage-2 Fine-tunings Weight: 🤗[Hugging Face](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main) / [Baidu]()

Please download them to your own machine. The file structure should appear as follows:
```
accessory/checkpoints/sphinx-v/stage2
├── consolidated.00-of-02.model.pth
├── consolidated.01-of-02.model.pth
├── tokenizer.model
├── config.json
└── meta.json
```

## 📁 Dataset
The all datasets for Stage-1(pre-training) and Stage-2(fine-tuning) can be found in [Dataset preparation](./Data/dataset.md).

**MDVP-Data**: 🤗[Hugging Face](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K)

**MDVP-Bench**: 🤗[Hugging Face](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K)


## 🚀 Training 

- **Prepare data**
  - Please download the annotations of our pre-training data and download the images from public open-source datasets. (Refer to the [Dataset preparation](./Data/dataset.md))

- **Stage1: Image-Visual Prompt-Text Alignment Pre-training**
  - Please download our pretrained SPHINX-v2-1k from [Hugging face](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/SPHINX/SPHINX-v2-1k)/[Baidu](https://pan.baidu.com/s/1PKCf515EGmSnSZ8teERHjQ?pwd=88z0)(提取码：88z0) After downloading, place the model in the "accessory/checkpoints/sphinx-v2-1k" directory.

  - You can find the pre-training configuration at [vp_pretrain.yaml](./accessory/config/data/vp_pretrain.yaml). Please ensure that all annotations are included and update the image paths in each JSON file to reflect the paths on your machine.

  - Set the relevant model path in the run script.

  - Start the pre-training process by running the `bash scripts/train_sphinx-v_pretrain_stage1.sh`.

- **Stage2: Multi-Task End-to-End Supervised Finetuning**
  - Download SPHINX-V Stage-1 Pre-training Weights from [Hugging Face](https://huggingface.co/sunshine-lwt/Osprey-7b/tree/main) or [Baidu](). Alternatively, you may use your own model weights trained from Stage 1.

  - You can find the fine-tuning configuration at [vp_finetune.yaml](./accessory/config/data/vp_finetune.yaml). Please ensure that all annotations are included and update the image paths in each JSON file to reflect the paths on your machine.

  - Set the relevant model path in the run script.
  
  - Run `bash scripts/train_sphinx-v_finetune_stage2.sh`.



## 📈 Evaluation 
See [evaluation](./accessory/eval/README.md) for details.


## 💌 Acknowledgement
- [LLaMA-Accessory](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
- [SAM](https://github.com/facebookresearch/segment-anything): the demo also uses the segmentation result from SAM.


## 🖊️: Citation

If you find **Draw-and-Understand** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{zhang2024mathverse,
  title={MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?},
  author={Zhang, Renrui and Jiang, Dongzhi and Zhang, Yichi and Lin, Haokun and Guo, Ziyu and Qiu, Pengshuo and Zhou, Aojun and Lu, Pan and Chang, Kai-Wei and Gao, Peng and others},
  journal={arXiv preprint arXiv:2403.14624},
  year={2024}
}
```