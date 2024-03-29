"""
Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_gpt_review.py
"""

import argparse
import json
import os
import pdb
import openai
import time
from tqdm import tqdm
import requests
import time
import json
from json.decoder import JSONDecodeError
import cv2
import numpy as np
from paint_util import paint_text_point, paint_text_box, encode_image

def get_eval(content: str, max_tokens: int):
    while True:
        try:
            messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }]
            ##########

            openai_api_key = ""
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }
            data = {
                "model": "gpt-4-vision-preview",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens":max_tokens
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            ret = result['choices'][0]['message']['content']

            ##########
            break

        except Exception as e:
            print(e)
        time.sleep(1)

    return ret


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


def main(args):
    phase = args.phase # android_QA_box
    if "_box" in args.output_path:
        index = args.output_path.find("_box")
    elif "_point" in args.output_path:
        index = args.output_path.find("_point")
    domain = args.output_path[:index] # android_QA

    if "natural" in phase: 
        context_str = "The image is a natural image."
    elif "ocr" in phase:
        context_str = "The image contains text, and the user wishes to know the content of the text."
    elif "screen" in phase:
        context_str = "The image is a screenshot from a mobile phone or webpage."
    elif "panel" in phase:
        context_str = "The image is a multi-panel figure."
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    
    question_path = f"mdvp_for_gpt4v_eval/{phase}/question.json"
    parser.add_argument('--question', default=question_path, help='path to question file')
    
    answer_list_path = [f"mdvp_for_gpt4v_eval/{phase}/answer.json", f"mdvp_for_gpt4v_eval/{phase}/prediction.json"]
    parser.add_argument('--answer-list', nargs='+', default=answer_list_path, help='gpt answer and model answer json files')
   
    rule_path = f"rule.json"
    parser.add_argument('--rule', default=rule_path ,help='gpt rule')
    

    parser.add_argument('--output', default=f"result/gpt_score.jsonl" ,help='output json dir')
    

    parser.add_argument('--max-tokens', type=int, default=2048, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    f_q = json.load(open(os.path.expanduser(args.question)))
    f_ans1 = json.load(open(os.path.expanduser(args.answer_list[0])))
    f_ans2 = json.load(open(os.path.expanduser(args.answer_list[1])))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    os.makedirs('./result', exist_ok=True)

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    handles = []
    idx = 0
    for ques, ans1, ans2 in tqdm(zip(f_q, f_ans1, f_ans2)):
        # paint som mark on image
        image_name = ques['image']
        image_path = f"/MDVP-bench/{domain}/" + image_name
        print("loading image from {}".format(image_path))
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        img_dimensions = (width, height) 
        if 'bbox' in ques['annotation']:
            bbox = ques['annotation']['bbox']
            paint_image_path = paint_text_box(image_path, bbox)
            rule = rule_dict["box"]
        elif 'points' in ques['annotation']:
            points = ques['annotation']['points']
            paint_image_path = paint_text_point(image_path, points)
            rule = rule_dict["point"]
        base64_image = encode_image(paint_image_path)
                    
        prompt = rule['prompt']
        role = rule['role']
        content_text = (f'[Context]\{context_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')

        content = [
                    {
                        "type": "text",
                        "text": content_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
        ]

        
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['question_id']),
            'category': phase
        }
        # pdb.set_trace()
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            print(review)
 
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            cur_js['answer1'] = ans1["text"]
            cur_js['answer2'] = ans2["text"]
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')

        idx += 1
        print(idx)
        
    review_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('--phase', help='MDVP domain') # android_QA_box
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()
    main(args)

    