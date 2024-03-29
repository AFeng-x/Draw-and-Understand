export output_results="path to refCOCOg brief caption outputs"

python transfer_to_cococap_format.py \
--output_results $output_results

python eval_cococap.py

python summarize_gpt_score.py --dir result