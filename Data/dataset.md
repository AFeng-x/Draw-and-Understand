## Data

### Stage 1 Pre-training Dataset

- 1. Please download the transformed annotations of each dataset from [Stage-1 Training Annotations]().

- 2. Please download the image from the official source.
| Data | Source |
| --- |  ---: |
| COCO 2014 | [Download](http://images.cocodataset.org/zips/train2014.zip) |
| COCO 2017 | [Download](http://images.cocodataset.org/zips/train2017.zip) |
| Visual Genome | [Download](https://opendatalab.com/OpenDataLab/Visual_Genome_Dataset_V1_dot_2) |
| Object365 | [Download](https://www.objects365.org/) |
| OpenImage | [Download](https://storage.googleapis.com/openimages/web/download_v7.html) |
| V3Det | [Download](https://v3det.openxlab.org.cn/) |
| ADE20k | [Download](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) |
| Cityscapes | [Download](https://www.cityscapes-dataset.com/downloads/) |
| cocostuff 10k | [Download](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip) |
| cocostuff 164k | [Download](https://github.com/nightrome/cocostuff#downloads) |
| VOCdevkit | [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar) |

| DocBank | [Download](https://doc-analysis.github.io/docbank-page/) |
| DocLayNet | [Download](https://huggingface.co/datasets/ds4sd/DocLayNet) |
| PubLayNet | [Download](https://developer.ibm.com/exchanges/data/all/publaynet/) |
| CurvedSynText150k | [Download](https://github.com/aim-uofa/AdelaiDet/blob/master/datasets/README.md) |
| ICDAR2013 | [Download](https://rrc.cvc.uab.es/?ch=2) |
| ICDAR2015 | [Download](https://drive.google.com/file/d/1J94245rU-s7KTecNQRD3KXG04ICZhL9z/view?usp=sharing) |
| MLT2017 | [Download](https://universityofadelaide.box.com/s/qu2wctdcsxh73bb94krdredpmx9nzf8m) |
| MLT2019 | [Download](https://rrc.cvc.uab.es/?ch=15&com=downloads) |
| TotalText | [Download](https://github.com/cs-chan/Total-Text-Dataset) |
| AITW | [Download](https://github.com/google-research/google-research/tree/master/android_in_the_wild) |

**Important notice**: `Visual Genome` should contain all the vg images(VG_100K and VG_100K_2). Merge the image data from the VG_100K and VG_100K_2 folders into one.

- 3. In each annotation JSON file, update the image path to reflect the location of the downloaded image data.


### Stage 2 Fine-tuning Dataset

- 1. Please download the transformed annotations of each dataset from [Stage-2 Training Annotations]().

- 2. Please download the image from the official source. The data for stages beyond stage 1 is list below:

| Data | Source |
| --- |  ---: |
| OpenPsgGCG | [Download](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/datasets.md) |
| GRIT | [Download](https://huggingface.co/datasets/zzliang/GRIT) |
| Flicker30K | [Download](https://shannon.cs.illinois.edu/DenotationGraph/) |
| M6Doc | [Download](https://github.com/HCIILAB/M6Doc/tree/main) |
| VCR | [Download](https://visualcommonsense.com/download/) |
| SeeClick | [Download](https://github.com/njucckevin/SeeClick/tree/main) |
| Multi-Panel | [Download]() |
| Osprey-724K | [Download](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K) |
| LaionGPT4v | [Download](https://huggingface.co/datasets/laion/gpt4v-dataset) |
| ShareGPT4v | [Download](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) |

- 3. In each annotation JSON file, update the image path to reflect the location of the downloaded image data.

<!-- **Important notice**:  -->



