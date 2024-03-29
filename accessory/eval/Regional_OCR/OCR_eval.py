import json
import re
import argparse

def extract_region_texts(annotation):
    """
    Extracts texts from gt annotation.
    Returns a list of categories.
    """
    regions = annotation.split('\n')
    texts = []

    for region in regions:
        parts = region.split(': ')
        if len(parts) > 1:
            texts.append(parts[1])
    
    return texts

def extract_text_in_parentheses(answer):
    """
    Extracts texts from model output.
    Returns a list of categories.
    """
    pattern = r"Region \d+: .*?\((.*?)\)\n|Region \d+: (.*?)\n"
    matches = re.findall(pattern, answer)
    result = []
    for match in matches:
        key_string = next(filter(None, match), 'BLANK')
        result.append(key_string if key_string else 'BLANK') 
    return result

def main():
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--output_results', type=str, required=True, help='Path to output results')
    args = parser.parse_args()

    with open(args.output_results, "r") as f:
        data = json.load(f)

    sum_count = 0
    match_count = 0
    for item in data:
        pred_list = extract_text_in_parentheses(item['answer'])
        gt_list = extract_region_texts(item['annotation'])
        assert(len(pred_list) == len(gt_list))
        sum_count += len(pred_list)
        for i in range(len(gt_list)):
            if pred_list[i].lower() == gt_list[i].lower():
                match_count += 1
        print(f"acc : {match_count/sum_count}")
    print(f"overall acc : {match_count/sum_count}")