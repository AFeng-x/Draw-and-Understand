export output_results="path to 80 detailed caption"

python transfer_to_gpt_format.py \
--output_path $output_results

python eval_gpt.py \
--question description/questions.json \
--answer-list description/answers.json \
              description/predictions.json \
--rule description/rule.json \
--output result/gpt_score.jsonl

python summarize_gpt_score.py --dir result