import argparse
import torch
import os
import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, CLIPImageProcessor
from pycocotools import mask as maskUtils
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import argparse
import re
import pdb

def extract_categories_from_string(string):
    """
    Extracts categories from a string where each line follows the pattern "Region X: category".
    Returns a list of categories.
    """
    categories = [line.split(": ")[1] for line in string.strip().split("\n") if ": " in line]
    return categories

def SemanticIOU(value: list[str], target: list[str]) -> None:

    intersection = len(set(value.split()) & set(target.split()))
    union = len(set(value.split()) | set(target.split()))

    return intersection / union

def main(args):
    if args.dataset in ["LVIS", "PACO"]:
        bert_model = SentenceTransformer(args.bert_model)
        with open(args.output_results, "r") as f:
            data_all = json.load(f)
        all_sim = 0
        all_num = 0
        all_iou = 0
        for data in tqdm(data_all): 
            gt_list = extract_categories_from_string(data["answer"])
            output_list = extract_categories_from_string(data["prediction"])
            for i in range(len(output_list)): 
                outputs = output_list[i]
                category = gt_list[i]
                category = category.replace('_', ' ')
                category = category.replace(':', ' ')
                print("[prediction]: ", outputs) 
                print("[gt category]:", category) 
                outputs_embeddings = bert_model.encode(outputs, convert_to_tensor=True)
                class_sentence_embeddings = bert_model.encode(category, convert_to_tensor=True)
                cosine_scores = util.cos_sim(outputs_embeddings, class_sentence_embeddings)

                semantic_iou = SemanticIOU(outputs.lower(), category.lower())

                all_sim += cosine_scores[0][0]
                all_iou += semantic_iou
                all_num += 1
            print("sim:{}, iou:{}".format(all_sim/all_num, all_iou/all_num))
        print("final sim:{}, semantic iou:{}".format(all_sim/all_num, all_iou/all_num))
    else:
        print(f"Unkown Dataset!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--bert_model', type=str, required=True, help='Path to the BERT model')
    parser.add_argument('--output_results', type=str, required=True, help='Path to output results')
    parser.add_argument('--dataset', type=str, required=True, help='PACO or LVIS')
    args = parser.parse_args()

    main(args)