<div align="center">

<img src="assets/figures/sphinx-v_text.png" style="width: 35%" alt="SPHINX-V Logo"/>

</div>

<div align="center">

## 🎨 Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want

[Weifeng Lin](), [Xinyu Wei](), [Ruichuan An](), [Peng Gao](), [Bocheng Zou](), [Yulin Luo](), [Siyuan Huang](), [Shanghang Zhang]() and [Hongsheng Li]()

</div>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://dreamllm.github.io/) [![arXiv Paper](https://img.shields.io/badge/arxiv-2312.10032-ECA8A7?logo=arxiv)](https://arxiv.org/pdf/2312.10032.pdf) [![Static Badge](https://img.shields.io/badge/Demo-6B88E3?logo=youtubegaming&logoColor=DAE4EE)](http://111.0.123.204:8000/) [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-blue.svg)](https://github.com/RunpeiDong/ChatDreamer-Private/blob/master/LICENSE)

[[🌐 Webpage](https://mathverse-cuhk.github.io/)] [[📖 Paper](https://arxiv.org/pdf/2403.14624.pdf)] [[🤗 MDVP-Data](https://huggingface.co/datasets/AI4Math/MathVerse)] [[🤗 MDVP-Bench](https://huggingface.co/datasets/AI4Math/MathVerse)] [[🤖️ Model](https://huggingface.co/datasets/AI4Math/MathVerse)] [[🎮 Demo](https://mathverse-cuhk.github.io/#leaderboard)]

</div>

## 💥 News

- **[2024.03.28]** 🚀 We release the [arXiv paper](https://arxiv.org/pdf/2403.14624) and some data samples in the [Webpage](https://mathverse-cuhk.github.io/#visualization).
<!-- - **[2024.03.22]** 🎉 **MathVerse** has been selected as 🤗 [***Hugging Face Daily Papers***](https://huggingface.co/papers/2403.14624)! -->


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


## 🚀 Ability Show

<details>
<summary>🔍 Natural Image Domain</summary>

<p align="center">
    <img src="figs/ver1.png" width="50%"> <br>
</p>
</details>

<details>
<summary>🔍 OCR Image Domain</summary>

<p align="center">
    <img src="figs/ver2.png" width="50%"> <br>
</p>
</details>

<details>
<summary>🔍 Mobile/Website Screenshot Domain</summary>

<p align="center">
    <img src="figs/ver3.png" width="50%"> <br>
</p>
</details>

<details>
<summary>🔍 Multi-panel Image Domain</summary>

<p align="center">
    <img src="figs/ver4.png" width="50%"> <br>
</p>
</details>
