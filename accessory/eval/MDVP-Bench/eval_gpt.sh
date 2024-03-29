export output_results="path to mdvp prediction results"

python transfer_to_gpt_format.py \
--output_path $output_results

python eval_gpt.py \
--phase android_QA_box # {mdvp domain}_{box/point}

python summarize_gpt_score.py --dir result