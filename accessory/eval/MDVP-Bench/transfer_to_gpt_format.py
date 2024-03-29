import json
import re
import argparse
import pdb
import os
'''
question:
{
    "question_id": 1, 
    "image": "000000104486.jpg", 
    "category": "natural_box", 
    "text": "Please analyze the relationship between all marked regions in the image.", 
    "annotation": {
        "bbox": [[157.23, 341.07, 10.67, 2.08]], 
        "segmentation": []
    }
}
answer:
{
    "question_id": 1, 
    "image": "000000104486.jpg", 
    "category": "natural_box", 
    "text": "<Region 1>: This region includes an individual who is caught in a moment that seems to involve some sort of task or activity. The person is engaged with a luggage cart, which suggests they might be arriving or departing from a location that offers such amenities, possibly a hotel. The cart holds luggage indicating travel or transit. The man's expression and attire provide clues to his role or state at the moment, such as potentially being a guest handling his luggage. The other individual seen partially in the background creates a sense of movement or interaction, but their relationship to the man or the context is unclear.\n"
}
predictions:
{
    "question_id": 1, 
    "image": "000000104486.jpg", 
    "category": "natural_box", 
    "text": "<Region 1>: The marked region does not appear to have any direct relationship with other marked regions, as there are no other marks to compare or contrast with.\n"
}
'''
def main(args):
    output_name = args.output_path.split("/")[-1] # android_QA_box.json
    phase = output_name.split(".")[0] # android_QA_box
    
    if "_box" in args.output_path:
        index = args.output_path.find("_box")
        vp = "bbox"
    elif "_point" in args.output_path:
        index = args.output_path.find("_point")
        vp = "points"
    domain = args.output_path[:index] # android_QA

    if not os.path.exists(f"mdvp_for_gpt4v_eval/{phase}"):
        os.mkdir(f"mdvp_for_gpt4v_eval/{phase}")

    with open(args.output_path, "r") as f:
        data = json.load(f)
    format_answer_list = []
    format_prediction_list = []
    for index, item in enumerate(data):
        format_answer = {
            "question_id": index+1, 
            "image": item["image_name"], 
            "category": phase, 
            "text": item["answer"]
        }
        format_answer_list.append(format_answer)

        format_prediction = {
            "question_id": index+1,
            "image": item["image_name"], 
            "category": phase, 
            "text": item["prediction"]
        }
        format_prediction_list.append(format_prediction)

    with open(f"mdvp_for_gpt4v_eval/{phase}/answer.json", "w") as f:
        json.dump(format_answer_list, f)
    print(f"mdvp_for_gpt4v_eval/{phase}/answer.json saved successfully!")  

    with open(f"mdvp_for_gpt4v_eval/{phase}/prediction.json", "w") as f:
        json.dump(format_prediction_list, f)
    print(f"mdvp_for_gpt4v_eval/{phase}/prediction.json saved successfully!")


    with open(f"MDVP-bench/{domain}/{output_name}", "r") as f:
        data = json.load(f)
    format_question_list = []
    for index, item in enumerate(data):
        format_question = {
            "question_id": index + 1, 
            "image": item["image_name"], 
            "category": phase, 
            "text": item["question"], 
            "annotation": {
                f"{vp}": item[f"{vp}"], 
                "segmentation": []
            }
        }
        format_question_list.append(format_question)
    with open(f"mdvp_for_gpt4v_eval/{phase}/question.json", "w") as f:
        json.dump(format_question_list, f)
    print(f"mdvp_for_gpt4v_eval/{phase}/question.json saved successfully!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to output results')
    args = parser.parse_args()

    main(args)