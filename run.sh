# "${1}": path to input file.
# "${2}": path to output file.

# python3.9 preprocess_data.py --train_file ${1} --test_file ${2}

### predict ###
python3.9 run_summarization.py \
    --model_name_or_path ./hw3_model \
    --do_predict \
    --test_file ${1} \
    --source_prefix "summarize: " \
    --output_dir ./tmp/predict \
    --output_file ${2} \
    --overwrite_output_dir \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --generation_num_beams 4

### eval_tw_rouge ###
# python3.9 ./ADL22-HW3/eval.py -r ${1} -s ${2}