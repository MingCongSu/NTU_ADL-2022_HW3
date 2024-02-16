# ADL22-HW3
Repository for ADL 2022 homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Download model
Use download.sh to download my fine-tuned model
```
bash download.sh
```
## metrics: tw_rouge
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
```

## Usage
### Train
```
python run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --source_prefix "summarize: " \
    --output_dir ./tmp/tst-summarization \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --num_train_epochs 3 \
    --save_strategy "no" \
    --save_total_limit 1 \
    --evaluation_strategy "epoch" \
```

### validation
```
python run_summarization.py \
    --model_name_or_path ./tmp/beam_search \
    --do_eval \
    --validation_file ./data/public.jsonl \
    --source_prefix "summarize: " \
    --output_dir ./tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --generation_num_beams 4 \
```

### Test/Predict
```
python run_summarization.py \
    --model_name_or_path ./tmp/beam_search \
    --do_predict \
    --test_file ./data/public.jsonl \
    --source_prefix "summarize: " \
    --output_dir ./tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_eval_batch_size=1 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --generation_num_beams 4
```

### Eval using tw_rouge
```
python ./ADL22-HW3/eval.py -r ./data/public.jsonl -s ./tmp/tst-summarization/predict/submission.jsonl
```

## Reference
[example/pytorch/summariztion](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)
[cccntu/tw_rouge](https://github.com/cccntu/tw_rouge)
