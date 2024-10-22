from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import pylab
import argparse
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder

def main():
    annFile = "annotations/RefCOCOg_annotation.json"
    resFile = ""
    
    with open(annFile, "r") as f:
        gts = json.load(f)
    with open(resFile, "r") as f:
        res = json.load(f)

    cocoEval = COCOEvalCap()

    cocoEval.evaluate(gts,res)

if __name__ == '__main__':
    main()