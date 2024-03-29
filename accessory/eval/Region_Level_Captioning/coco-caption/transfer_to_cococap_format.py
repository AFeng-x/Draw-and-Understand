import json
import re
import argparse
import pdb

def extract_region_descriptions(text):
    pattern = r'Region \d+: (.*?)\n'
    descriptions = re.findall(pattern, text)
    return descriptions

'''
annFile:
"1": [{"image_id": 1, "id": 1, "caption": "PART OF BLACK CHAIR LEG"}]

resFile:
"1": [{"image_id": 1, "id": 1, "caption": "PART OF BLACK CHAIR LEG"}]
'''
with open("../RefCOCOg_brief_caption.json", "r") as f:
    data = json.load(f)
ann_dict = {}
for index, item in enumerate(data):
    ann_dict[index+1] = [{
        "image_id": index+1,
        "id": index+1,
        "caption": extract_region_descriptions(item["gt_answers"])
    }]
pdb.set_trace()
with open("annotations/RefCOCOg_annotation.json", "w") as f:
    json.dump(ann_dict, f)
pdb.set_trace()

def main(args):
    with open(args.output_results, "r") as f:
        data = json.load(f)

    anno_dict = {}

    for index, item in enumerate(data):
        ann_dict[index+1] = [{
            "image_id": index+1,
            "id": index+1,
            "caption": extract_region_descriptions(item["prediction"])
        }]


    with open(f"results/RefCOCOg.json", "w") as f:
        json.dump(anno_dict, f)
    print(f"results/RefCOCOg.json saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--output_results', type=str, required=True, help='Path to output results')
    args = parser.parse_args()

    main(args)