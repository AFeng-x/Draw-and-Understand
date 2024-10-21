# Evaluation

## Referring Object Classification

- **Data Preparation**

Download evaluation data from [🤗HF](https://huggingface.co/datasets/Afeng-x/Draw-and-Understand/tree/main/eval_dataset/Referring_Object_Classification) and save to `Referring_Object_Classification` directory. Your  `Referring_Object_Classification` directory look like:

```css
Referring_Object_Classification/
│
├── LVIS_box.json
├── LVIS_point.json
├── PACO_box.json
├── PACO_point.json
├── ROC_eval.py
└── ROC_eval.sh
```

Images for LVIS and PACO datasets are sourced from COCO 2017. You can download the images from the [COCO website](https://cocodataset.org/#download). Then, modify the `--img_root` parameter in `infer_and_save.sh` to point to the directory where the COCO images are located:

```shell
CUDA_VISIBLE_DEVICES=${GPU} PYTHONWARNINGS="ignore" python -W ignore eval.py \
--dataset COCOText_box \
--llama_type llama_ens5_vp \
--llama_config ${llama_config} \
--tokenizer_path ${tokenizer_path} \
--pretrained_path ${pretrained_path} \
--img_root "path to coco images" \
--batch_size 2 \
--model_parallel_size 1
```

Next, you can modify the `--dataset` parameter to 'LVIS_box', 'LVIS_point', 'PACO_box', or 'PACO_point' to perform evaluation for each type.

If you want to test your own benchmark, organize your data in the following format:

```json
{
  "image_name": "000000490182.jpg", 
  "question": "Please identify the labels of each marked region in the image.", 
  "question_id": 490182, 
  "dataset_name": "LVIS_box", 
  "gt_answers": "Region 1: baseball_base\nRegion 2: baseball_glove\nRegion 3: belt_buckle\nRegion 4: belt\nRegion 5: belt", 
  "bbox": [
    [292.48, 352.59, 95.01, 24.52], 
    [401.27, 128.58, 24.94, 36.22], 
    [201.95, 166.28, 14.04, 9.48], 
    [180.56, 160.87, 47.61, 14.71], 
    [312.48, 219.3, 38.92, 14.8]
  ]
}
```

Then, declare your dataset in `vp_annotation_config.json` and specify `max_token`:

```json
"LVIS_box": {
        "test": "eval/Referring_Object_Classification/LVIS_box.json",
        "max_new_tokens": 256
    }
```

Inference results will be saved in the `./results/` directory as `{dataset}_results.json` files.

- **Start eval**

Use the generated `{dataset}_results.json` file for evaluation:

```shell
cd Referring_Object_Classification
python ROC_eval.py --bert_model "path to all-MiniLM-L6-v2" --output_results "path to {dataset}.json" --dataset "{PACO/LVIS}"
```

## Region Level Captioning

- **Data preparation**

Download evaluation data from [🤗HF](https://huggingface.co/datasets/Afeng-x/Draw-and-Understand/tree/main/eval_dataset/Region_Level_Captioning) and save to `Region_Level_Captioning` directory. Your  `Region_Level_Captioning` directory look like:

```css
Region_Level_Captioning/
│
├── coco-caption/
│   ├── annotations
│   └── ...
│
└── Osprey_80_detail_caption.json
└── RefCOCOg_brief_caption.json
```

Download pycocoevalcap (for sphinx-v) from [Google Drive](https://drive.google.com/file/d/1PxH_1E2xSLAqwjn7bdM8qmRnp30lqKq3/view?usp=sharing) or [Baidu NetDisk](https://pan.baidu.com/s/1Y5LAagvYmI4Gtobrdc22iw?pwd=bn0g). Save it to `Region_Level_Captioning/pycocoevalcap`.

We use RefCOCOg for brief caption testing, with METEOR and CIDEr as metrics. Images in RefCOCOg are sourced from COCO 2014. Additionally, we follow [Osprey](https://github.com/circleradon/osprey)'s approach by using 80 images from COCO 2017 for detailed caption testing, evaluating the quality of results using GPT-4.

Modify the '--dataset' parameter in `infer_and_save.sh` to either 'RefCOCOg_brief_caption' or 'Osprey_80_detail_caption' for respective inference.

- **Start Eval**

Perform brief caption evaluation using the generated `{dataset}_results.json` file:

```shell
cd Region_Level_Captioning/coco-caption
sh eval_cococap.sh
```

Or conduct detailed caption evaluation:

```shell
cd Region_Level_Captioning/coco-caption
sh eval_gpt.sh
```

## Regional OCR

- **Data preparation**

Download evaluation data from [🤗HF](https://huggingface.co/datasets/Afeng-x/Draw-and-Understand/tree/main/eval_dataset/Regional_OCR) and save to `Regional_OCR` directory. Your  `Regional_OCR` directory look like:

```css
Regional_OCR/
│
├── COCOText_box.json
└── OCR_eval.py
```

The images in COCOText are also sourced from COCO 2017.

Generate the `COCOText_results.json` file by running the model inference.

- **Start Eval**

```shell
cd Regional_OCR
python OCR_eval.py --output_results "path to COCOText_results.json"
```

## MDVP-Bench

- **Data preparation**

Download evaluation data from [🤗HF](https://huggingface.co/datasets/Afeng-x/Draw-and-Understand/tree/main/MDVP-bench) and save to `MDVP-Bench`directory. Your  `MDVP-Bench` directory look like:

```css
MDVP-Bench/
│
├── mdvp_for_gpt4v_eval/
│
└── MDVP-bench/
│   ├── android_detailed_caption/
│   └── android_QA/
│   └── ...
└── ...
```

MDVP-Bench is divided into multiple domains, each domain further divided into point and box visual prompts.

Modify the '--dataset' parameter in `infer_and_save.sh` to '{mdvp domain}_{box/point}' for respective inference.

- **Start Eval**

Perform brief caption evaluation using the generated '{mdvp domain}_{box/point}_results.json' file:

```shell
cd MDVP-Bench
sh eval_gpt.sh
```
