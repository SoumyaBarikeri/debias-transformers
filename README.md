## Quick tour - Debias DialoGPT by Transfer learning 

As part of this thesis work DialoGPT is debiased for 5 demographics - Religion1 (Jews-Christians), Religion2 (Muslims-Christians), Race (African-American), Gender(Female-Male) and Sexual orientation (LGBTQ-Straight).

Below are the commands to carry out Algorithmic level and Data level Debiasing in pre-trained DialoGPT model. Examples are shown only for the demographic - Religion1 (Jews-Christains). In case of any other demographic, change the demographic fileds and data files accordingly.

Note: Debiasing scripts are found in the path debias_transformers/examples/language-modeling/. The data required all the below commands are found in https://github.com/SoumyaBarikeri/RedditBias/tree/master/data and https://github.com/SoumyaBarikeri/RedditBias/tree/master/text_files .

Note: The debiased models are found in dws-09 server at /work-ceph/sbariker/models/DEMOGRAPHIC_NAME/ -> Replace DEMOGRAPHIC_NAME with specific demographic like religion1, religion2, race, gender or orientation.

### Algoritmic level Debiasing - Equalising loss over per sentence Target pairs

```python
CUDA_VISIBLE_DEVICES=1 python debias_lm_grid.py \
    --output_dir=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --model_type=gpt2 \
    --model_name_or_path=microsoft/DialoGPT-small \
    --config_name=microsoft/DialoGPT-small \
    --tokenizer_name=microsoft/DialoGPT-small \
    --save_total_limit=2 \
    --num_train_epochs=2.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=2000 \
    --save_steps=2000 \
    --train_data_file=/work-ceph/sbariker/data/text_files/religion1/religion1_bias_manual_train.txt \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/data/text_files/humanref6k.txt \
    --block_size=36 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir \
    --debiasing_head=EqualisingLoss \
    --debias_method=EqualisingLoss \
    --embedding_type=output \
    --demographic=religion1 \
    --target_pair_type=per_sent_targets \
    --norm_debias_loss \
    --demo1_valid=/work-ceph/sbariker/data/text_files/religion1/religion1_jews_biased_valid_reduced.txt \
    --demo2_valid=/work-ceph/sbariker/data/text_files/religion1/religion1_christians_biased_valid_reduced.txt \
    --demo1_test=/work-ceph/sbariker/data/text_files/religion1/religion1_jews_biased_test_reduced.txt \
    --demo2_test=/work-ceph/sbariker/data/text_files/religion1/religion1_christians_biased_test_reduced.txt
```
### Algoritmic level Debiasing - Cosine Distance equalising loss

```python
CUDA_VISIBLE_DEVICES=1 python debias_lm_grid.py \
    --output_dir=/work-ceph/sbariker/models/religion1/cos_loss_grid/ \
    --model_type=gpt2 \
    --model_name_or_path=microsoft/DialoGPT-small \
    --config_name=microsoft/DialoGPT-small \
    --tokenizer_name=microsoft/DialoGPT-small \
    --save_total_limit=2 \
    --num_train_epochs=2.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=2000 \
    --save_steps=2000 \
    --train_data_file=/work-ceph/sbariker/data/text_files/religion1/religion1_bias_manual_train.txt \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/data/text_files/humanref6k.txt \
    --block_size=36 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir \
    --debiasing_head=Cosine \
    --debias_method=Cosine \
    --demographic=religion1 \
    --embedding_type=output \
    --demo1_valid=/work-ceph/sbariker/data/text_files/religion1/religion1_jews_biased_valid_reduced.txt \
    --demo2_valid=/work-ceph/sbariker/data/text_files/religion1/religion1_christians_biased_valid_reduced.txt \
    --demo1_test=/work-ceph/sbariker/data/text_files/religion1/religion1_jews_biased_test_reduced.txt \
    --demo2_test=/work-ceph/sbariker/data/text_files/religion1/religion1_christians_biased_test_reduced.txt 
```
### Algoritmic level Debiasing - Projection based Hard debiasing loss

```python
CUDA_VISIBLE_DEVICES=1 python debias_lm_grid.py \
    --output_dir=/work-ceph/sbariker/models/religion1/hard_de_grid/ \
    --model_type=gpt2 \
    --model_name_or_path=microsoft/DialoGPT-small \
    --config_name=microsoft/DialoGPT-small \
    --tokenizer_name=microsoft/DialoGPT-small \
    --save_total_limit=2 \
    --num_train_epochs=2.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=2000 \
    --save_steps=2000 \
    --train_data_file=/work-ceph/sbariker/data/text_files/religion1/religion1_bias_manual_train.txt \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/data/text_files/humanref6k.txt \
    --block_size=36 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir \
    --debiasing_head=HardDe \
    --debias_method=HardDe \
    --demographic=religion1 \
    --embedding_type=output \
    --demo1_valid=/work-ceph/sbariker/data/text_files/religion1/religion1_jews_biased_valid_reduced.txt \
    --demo2_valid=/work-ceph/sbariker/data/text_files/religion1/religion1_christians_biased_valid_reduced.txt \
    --demo1_test=/work-ceph/sbariker/data/text_files/religion1/religion1_jews_biased_test_reduced.txt \
    --demo2_test=/work-ceph/sbariker/data/text_files/religion1/religion1_christians_biased_test_reduced.txt
```

### Data level debiasing - Counter Target Data Augmentation (CTDA)

```python
CUDA_VISIBLE_DEVICES=3 python run_language_modeling.py \
    --output_dir=/work-ceph/sbariker/models/religion1/lm_loss_swapped_target/ \
    --model_type=gpt2 \
    --model_name_or_path=microsoft/DialoGPT-small \
    --config_name=microsoft/DialoGPT-small \
    --tokenizer_name=microsoft/DialoGPT-small \
    --save_total_limit=2 \
    --num_train_epochs=2.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=2000 \
    --save_steps=2000 \
    --train_data_file=/work-ceph/sbariker/data/text_files/religion1/religion1_bias_manual_swapped_targets_train.txt \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/data/text_files/humanref6k.txt \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --block_size=36 \
    --gradient_accumulation_steps=1 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir
```

#### Evaluation: Significance results on testset - 

```python
python debias_transformers/evaluation/measure_bias_reduced_args.py     --data_path=/work-ceph/sbariker/data/     --log_path=/work-ceph/sbariker/logs/     --get_perp=yes     --save_perp=no     --demo=religion1     --demo1=jews     --demo2=christians     --input_file_1=reddit_comments_religion1_jews_processed_phrase_biased_testset_reduced.csv     --input_file_2=reddit_comments_religion1_christians_processed_phrase_biased_testset_reduced.csv     --model_path=/work-ceph/sbariker/models/religion1/lm_loss_swapped_target/    --model_name=lm_loss_swapped_target
```

### Data level debiasing - Counter Attribute Data Augmentation (CADA)

```python
CUDA_VISIBLE_DEVICES=0 python run_language_modeling.py \
    --output_dir=/work-ceph/sbariker/models/religion1/lm_loss_swapped_attr/ \
    --model_type=gpt2 \
    --model_name_or_path=microsoft/DialoGPT-small \
    --config_name=microsoft/DialoGPT-small \
    --tokenizer_name=microsoft/DialoGPT-small \
    --save_total_limit=2 \
    --num_train_epochs=2.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=2000 \
    --save_steps=2000 \
    --train_data_file=/work-ceph/sbariker/data/text_files/religion1/religion1_bias_manual_swapped_attr_train.txt \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/data/text_files/humanref6k.txt \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --block_size=36 \
    --gradient_accumulation_steps=1 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir
```

#### Evaluation: Significance results on testset - 

```python
python debias_transformers/evaluation/measure_bias_reduced_args.py     --data_path=/work-ceph/sbariker/data/     --log_path=/work-ceph/sbariker/logs/     --get_perp=yes     --save_perp=no     --demo=religion1     --demo1=jews     --demo2=christians     --input_file_1=reddit_comments_religion1_jews_processed_phrase_biased_testset_neg_attr_reduced.csv     --input_file_2=reddit_comments_religion1_jews_processed_phrase_unbiased_testset_pos_attr_reduced.csv     --model_path=/work-ceph/sbariker/models/religion1/lm_loss_swapped_attr/    --model_name=lm_loss_swapped_attr
```

## Quick tour - Evaluation of Debiased models on Dialog State Tracking (DST) task

Below command evaluates DialoGPT debiased on Demographic - Religion1, based on Equalising loss

Note: The cleaned MultiWoz2 data can be found at https://github.com/SoumyaBarikeri/RedditBias/tree/master/data/clean_data . The Script (lm_dst_binary.py) to fine-tune models on DST task is found in debias_transformers/examples/language-modeling

Note: The debiased models fine-tuned on DST are found in /work-ceph/sbariker/models/dst/ on dws-09 server

```python
CUDA_VISIBLE_DEVICES=0 python lm_dst_binary.py \
    --output_dir=/work-ceph/sbariker/models/dst/rel1_eq/ \
    --model_type=gpt2 \
    --model_name_or_path=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --config_name=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --tokenizer_name=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --save_total_limit=2 \
    --num_train_epochs=1.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=10000 \
    --save_steps=10000 \
    --train_data_file=/work-ceph/sbariker/data/multiwoz/clean_data/train_dials.json \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/data/multiwoz/clean_data/test_dials.json \
    --onto_file_path=/work-ceph/sbariker/data/multiwoz/ontology.json \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=12 \
    --block_size=128 \
    --gradient_accumulation_steps=4 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir \
    --label_names=dst_labels
```

## Quick tour - Evaluation of Debiased models on Dialog System Technology Challenge 7 (DSTC7) Response generation task

Below command evaluates response generation capability of DialoGPT debiased on Demographic - Religion1, based on Equalising loss

Note: The Script (lm_dstc7.py) to fine-tune models on DSTC 7 task is found in debias_transformers/examples/language-modeling. The training dataset train_convos.txt for DSTC7 can be found in dws-09 university server at /work-ceph/sbariker/DSTC7-End-to-End-Conversation-Modeling/data_extraction/data-official/ , additionally the train_convos.txt can be generated referring to DSTC7 github (https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction). The test set can be found at https://github.com/SoumyaBarikeri/RedditBias/tree/master/data/dstc7 or /work-ceph/sbariker/DSTC7-End-to-End-Conversation-Modeling/data_extraction/data-official-test/test_convos_processed.txt .

Note: The debiased models fine-tuned on DSTC7 data are found in /work-ceph/sbariker/models/dstc7/ on dws-09 server

```python
CUDA_VISIBLE_DEVICES=1 python lm_dstc7.py \
    --output_dir=/work-ceph/sbariker/models/dstc7/rel1_eq/ \
    --model_type=gpt2 \
    --model_name_or_path=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --config_name=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --tokenizer_name=/work-ceph/sbariker/models/religion1/eq_loss_grid/ \
    --save_total_limit=2 \
    --num_train_epochs=1 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=10000 \
    --save_steps=10000 \
    --train_data_file=/work-ceph/sbariker/DSTC7-End-to-End-Conversation-Modeling/data_extraction/data-official/train_convos.txt \
    --do_eval \
    --eval_data_file=/work-ceph/sbariker/DSTC7-End-to-End-Conversation-Modeling/data_extraction/data-official-test/test_convos_processed.txt \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps=5 \
    --line_by_line \
    --force_pad_token \
    --overwrite_output_dir \
    --output_resp_file=/work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp.txt

```

### Evaluation: 

#### 1. Generate test_convos.txt like file with '\_\_UNDISCLOSED\_\_' replaced with the model response

```python
python debias_transformers/evaluation/prepare_dstc7_response.py --hyp_file=/work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp.txt    --ref_file=/work-ceph/sbariker/DSTC7-End-to-End-Conversation-Modeling/data_extraction/data-official-test/test_convos.txt	--dest_file=/work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp_test_convos.txt
```
#### 2. Evaluate generated responses using dstc.py [script](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/dstc.py) provided by DSTC 7 team

Note: Clone the DSTC 7 [repository](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling) and run the below command. Also the DSTC 7 [data](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) should be generated beforehand.

```python
python dstc.py -c /work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp_test_convos.txt --refs ../../data_extraction/test.refs
```
