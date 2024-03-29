import json
import re
import argparse
import pdb
'''
{
    "question_id": 0, 
    "image": "000000057870.jpg", 
    "category": "description", 
    "text": "There is a wooden chair with its back facing towards us. This chair, the second from the left, is situated near a plant and almost seems to cover it. Similar to others in the room, it features a rattan rounded back and blue accents."
}
'''
def main(args):
    with open(args.output_path, "r") as f:
        data = json.load(f)
    result_list = []
    for index, item in enumerate(data):
        format_item = {
            "question_id": index,
            "image": item['image_name'],
            "categroy": "description",
            "text": item['prediction']
        }
        result_list.append(format_item)
        # pdb.set_trace()
    with open("description/predictions.json", "w") as f:
        json.dump(result_list, f)
    print(f"description/predictions.json saved successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to output results')
    args = parser.parse_args()

    main(args)