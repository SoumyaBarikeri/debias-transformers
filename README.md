<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0
</h3>

## Quick tour - Debias DialoGPT by Transfer learning 

As part of this thesis work DialoGPT is debiased for 5 demographics - Religion1 (Jews-Christians), Religion2 (Muslims-Christians), Race (African-American), Gender(Female-Male) and Sexual orientation (LGBTQ-Straight).

Below are the commands to carry out Algorithmic level and Data level Debiasing in pre-trained DialoGPT model. Examples are shown only for the demographic - Religion1 (Jews-Christains). In case of any other demographic, change the demographic fileds and data files accordingly.

Note: Debiasing scripts are found in the path debias_transformers/examples/language-modeling/

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
python ../../../evaluation/measure_bias_reduced_args.py     --data_path=/work-ceph/sbariker/data/     --log_path=/work-ceph/sbariker/logs/     --get_perp=yes     --save_perp=no     --demo=religion1     --demo1=jews     --demo2=christians     --input_file_1=reddit_comments_religion1_jews_processed_phrase_biased_testset_reduced.csv     --input_file_2=reddit_comments_religion1_christians_processed_phrase_biased_testset_reduced.csv     --model_path=/work-ceph/sbariker/models/religion1/lm_loss_swapped_target/    --model_name=lm_loss_swapped_target
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
python ../../../evaluation/measure_bias_reduced_args.py     --data_path=/work-ceph/sbariker/data/     --log_path=/work-ceph/sbariker/logs/     --get_perp=yes     --save_perp=no     --demo=religion1     --demo1=jews     --demo2=christians     --input_file_1=reddit_comments_religion1_jews_processed_phrase_biased_testset_neg_attr_reduced.csv     --input_file_2=reddit_comments_religion1_jews_processed_phrase_unbiased_testset_pos_attr_reduced.csv     --model_path=/work-ceph/sbariker/models/religion1/lm_loss_swapped_attr/    --model_name=lm_loss_swapped_attr
```

## Quick tour - Evaluation of Debiased models on Dialog State Tracking (DST) task

Below command evaluates DialoGPT debiased on Demographic - Religion1, based on Equalising loss

Note: The Script (lm_dst_binary.py) to fine-tune models on DST task is found in debias_transformers/examples/language-modeling

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

Note: The Script (lm_dstc7.py) to fine-tune models on DSTC 7 task is found in debias_transformers/examples/language-modeling

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
python evaluation/prepare_dstc7_response.py --hyp_file=/work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp.txt    --ref_file=/work-ceph/sbariker/DSTC7-End-to-End-Conversation-Modeling/data_extraction/data-official-test/test_convos.txt	--dest_file=/work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp_test_convos.txt
```
#### 2. Evaluate generated responses using dstc.py [script](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/dstc.py) provided by DSCT 7 team

Note: Clone the DSTC 7 [repository](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling) and run the below command. Also the DSTC 7 [data] (https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) should be generated beforehand.

```python
python dstc.py -c /work-ceph/sbariker/data/eval_dsct7/rel1_eq_resp_test_convos.txt --refs ../../data_extraction/test.refs
```


🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, etc in 100+ languages. Its aim is to make cutting-edge NLP easier to use for everyone. 

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets then share them with the community on our [model hub](https://huggingface.co/models). At the same time, each python module defining an architecture can be used as a standalone and modified to enable quick research experiments. 

🤗 Transformers is backed by the two most popular deep learning libraries, [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/), with a seamless integration between them, allowing you to train your models with one then load it for inference with the other.

### Recent contributors
[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/0)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/0)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/1)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/1)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/2)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/2)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/3)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/3)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/4)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/4)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/5)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/5)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/6)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/6)[![](https://sourcerer.io/fame/clmnt/huggingface/transformers/images/7)](https://sourcerer.io/fame/clmnt/huggingface/transformers/links/7)

## Online demos

You can test most of our models directly on their pages from the [model hub](https://huggingface.co/models). We also offer an [inference API](https://huggingface.co/pricing) to use those models.

Here are a few examples: 
- [Masked word completion with BERT](https://huggingface.co/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Name Entity Recognition with Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Text generation with GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+)
- [Natural Langugage Inference with RoBERTa](https://huggingface.co/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Summarization with BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Question answering with DistilBERT](https://huggingface.co/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Translation with T5](https://huggingface.co/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

**[Write With Transformer](https://transformer.huggingface.co)**, built by the Hugging Face team, is the official demo of this repo’s text generation capabilities.

## Quick tour

To immediately use a model on a given text, we provide the `pipeline` API. Pipelines group together a pretrained model with the preprocessing that was used during that model training. Here is how to quickly use a pipeline to classify positive versus negative texts 

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to include pipeline into the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9978193640708923}]
```

The second line of code downloads and caches the pretrained model used by the pipeline, the third line evaluates it on the given text. Here the answer is "positive" with a confidence of 99.8%. 

This is another example of pipeline used for that can extract question answers from some context:

``` python
>>> from transformers import pipeline

# Allocate a pipeline for question-answering
>>> question_answerer = pipeline('question-answering')
>>> question_answerer({
...     'question': 'What is the name of the repository ?',
...     'context': 'Pipeline have been included in the huggingface/transformers repository'
... })
{'score': 0.5135612454720828, 'start': 35, 'end': 59, 'answer': 'huggingface/transformers'}

```

On top of the answer, the pretrained model used here returned its confidence score, along with the start position and its end position in the tokenized sentence. You can learn more about the tasks supported by the `pipeline` API in [this tutorial](https://huggingface.co/transformers/task_summary.html).

To download and use any of the pretrained models on your given task, you just need to use those three lines of codes (PyTorch verison):
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = AutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
or for TensorFlow:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

The tokenizer is responsible for all the preprocessing the pretrained model expects, and can be called directly on one (or list) of texts (as we can see on the fourth line of both code examples). It will output a dictionary you can directly pass to your model (which is done on the fifth line).

The model itself is a regular [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) or a [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (depending on your backend) which you can use normally. For instance, [this tutorial](https://huggingface.co/transformers/training.html) explains how to integrate such a model in classic PyTorch or TensorFlow training loop, or how to use our `Trainer` API to quickly fine-tune the on a new dataset.

## Why should I use transformers?

1. Easy-to-use state-of-the-art models:
    - High performance on NLU and NLG tasks.
    - Low barrier to entry for educators and practitioners.
    - Few user-facing abastractions with just three classes to learn.
    - A unified API for using all our pretrained models.

1. Lower compute costs, smaller carbon footprint:
    - Researchers can share trained models instead of always retraining.
    - Practitioners can reduce compute time and production costs.
    - Dozens of architectures with over 2,000 pretrained models, some in more than 100 languages.

1. Choose the right framework for every part of a model's lifetime:
    - Train state-of-the-art models in 3 lines of code.
    - Move a single model between TF2.0/PyTorch frameworks at will.
    - Seamlessly pick the right framework for training, evaluation, production.

1. Easily customize a model or an example to your needs:
    - Examples for each architecture to reproduce the results by the official authors of said architecture.
    - Expose the models internal as consistently as possible.
    - Model files can be used independently of the library for quick experiments. 

## Why shouldn't I use transformers?

- This library is not a modular toolbox of building blocks for neural nets. The code in the model files is not refactored with additional abstractions on purpose, so that researchers can quickly iterate on each of the models without diving in additional abstractions/files.
- The training API is not intended to work on any model but is optimized to work with the models provided by the library. For generic machine learning loops, you should use another library.
- While we strive to present as many use cases as possible, the scripts in our [examples folder](https://github.com/huggingface/transformers/tree/master/examples) are just that: examples. It is expected that they won't work out-of-the box on your specific problem and that you will be required to change a few lines of code to adapt them to your needs.

## Installation

This repository is tested on Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for [examples](https://github.com/huggingface/transformers/tree/master/examples)) and TensorFlow 2.0.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then, you will need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```

If you'd like to play with the examples, you must [install the library from source](https://huggingface.co/transformers/installation.html#installing-from-source).

## Models architectures

🤗 Transformers currently provides the following architectures (see [here](https://huggingface.co/transformers/model_summary.html) for a high-level summary of each them):

1. **[BERT](https://huggingface.co/transformers/model_doc/bert.html)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. **[GPT](https://huggingface.co/transformers/model_doc/gpt.html)** (from OpenAI) released with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. **[GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)** (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. **[Transformer-XL](https://huggingface.co/transformers/model_doc/transformerxl.html)** (from Google/CMU) released with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. **[XLNet](https://huggingface.co/transformers/model_doc/xlnet.html)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. **[XLM](https://huggingface.co/transformers/model_doc/xlm.html)** (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
7. **[RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (from HuggingFace), released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/master/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/master/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
9. **[CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)** (from Salesforce) released with the paper [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
10. **[CamemBERT](https://huggingface.co/transformers/model_doc/camembert.html)** (from Inria/Facebook/Sorbonne) released with the paper [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
11. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
12. **[T5](https://huggingface.co/transformers/model_doc/t5.html)** (from Google AI) released with the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
13. **[XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)** (from Facebook AI), released together with the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
14. **[MMBT](https://github.com/facebookresearch/mmbt/)** (from Facebook), released together with the paper a [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/pdf/1909.02950.pdf) by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
15. **[FlauBERT](https://huggingface.co/transformers/model_doc/flaubert.html)** (from CNRS) released with the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
16. **[BART](https://huggingface.co/transformers/model_doc/bart.html)** (from Facebook) released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
17. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
18. **[DialoGPT](https://huggingface.co/transformers/model_doc/dialogpt.html)** (from Microsoft Research) released with the paper [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
19. **[Reformer](https://huggingface.co/transformers/model_doc/reformer.html)** (from Google Research) released with the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
20. **[MarianMT](https://huggingface.co/transformers/model_doc/marian.html)** Machine translation models trained using [OPUS](http://opus.nlpl.eu/) data by Jörg Tiedemann. The [Marian Framework](https://marian-nmt.github.io/) is being developed by the Microsoft Translator Team.
21. **[Longformer](https://huggingface.co/transformers/model_doc/longformer.html)** (from AllenAI) released with the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
22. **[DPR](https://github.com/facebookresearch/DPR)** (from Facebook) released with the paper [Dense Passage Retrieval
for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by Vladimir Karpukhin, Barlas Oğuz, Sewon
Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
23. **[Pegasus](https://github.com/google-research/pegasus)** (from Google) released with the paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)> by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu.
24. **[MBart](https://github.com/pytorch/fairseq/tree/master/examples/mbart)** (from Facebook) released with the paper  [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.  
25. **[LXMERT](https://github.com/airsplay/lxmert)** (from UNC Chapel Hill) released with the paper [LXMERT: Learning Cross-Modality Encoder Representations from Transformers for Open-Domain Question Answering](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal.
26. **[Funnel Transformer](https://github.com/laiguokun/Funnel-Transformer)** (from CMU/Google Brain) released with the paper [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
27. **[LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm)** (from Microsoft Research Asia) released with the paper [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
28. **[Other community models](https://huggingface.co/models)**, contributed by the [community](https://huggingface.co/users).
29. Want to contribute a new model? We have added a **detailed guide and templates** to guide you in the process of adding a new model. You can find them in the [`templates`](./templates) folder of the repository. Be sure to check the [contributing guidelines](./CONTRIBUTING.md) and contact the maintainers or open an issue to collect feedbacks before starting your PR.

These implementations have been tested on several datasets (see the example scripts) and should match the performances of the original implementations. You can find more details on the performances in the Examples section of the [documentation](https://huggingface.co/transformers/examples.html).


## Learn more

| Section | Description |
|-|-|
| [Documentation](https://huggingface.co/transformers/) | Full API documentation and tutorials |
| [Task summary](https://huggingface.co/transformers/task_summary.html) | Tasks supported by 🤗 Transformers |
| [Preprocessing tutorial](https://huggingface.co/transformers/preprocessing.html) | Using the `Tokenizer` class to prepare data for the models |
| [Training and fine-tuning](https://huggingface.co/transformers/training.html) | Using the models provided by 🤗 Transformers in a PyTorch/TensorFlow training loop and the `Trainer` API |
| [Quick tour: Fine-tuning/usage scripts](https://github.com/huggingface/transformers/tree/master/examples) | Example scripts for fine-tuning models on a wide range of tasks |
| [Model sharing and uploading](https://huggingface.co/transformers/model_sharing.html) | Upload and share your fine-tuned models with the community |
| [Migration](https://huggingface.co/transformers/migration.html) | Migrate to 🤗 Transformers from `pytorch-transformers` or `pytorch-pretrained-bert` |

## Citation

We now have a [paper](https://arxiv.org/abs/1910.03771) you can cite for the 🤗 Transformers library:
```bibtex
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```
