# Implementation for BLANC

This repository contains code for the paper "Context-Aware Answer Extraction in Question Answering".

## Build train/dev/test set from NaturalQ-MRQA dataset

1. Download NaturalQ (MRQA) train/dev dataset ([link](https://github.com/mrqa/MRQA-Shared-Task-2019#datasets)) and locate the train/dev set in data/naturalQ/ directory

2. Rename the dev set to test.jsonl.gz

3. Run split_data.py in code/src/preprocessor as follows:

```bash
python split_data.py --source ../../../data/naturalQ/train.jsonl.gz --train_output train.jsonl --dev_output dev.jsonl --data_type mrqa
```

4. Compress train.jsonl and dev.jsonl with gzip (gzip train.jsonl; gzip dev.jsonl)

5. Locate these two files into data/naturalQ/ directory

6. Now you have train/dev/test set in data/naturalQ directory


## Train BLANC on NaturalQ

Run BLANC script in code/ as follows:

```bash
LABEL=trial_001 GEOP=0.99 WINS=3 LMB=0.8 bash run_blanc_naturalqa.sh
```

## Test BLANC

Indicate which checkpoint you want to test.

In this case, we test BLANC with trial_001 checkpoint.

```bash
LABEL=trial_001 GEOP=0.99 WINS=3 LMB=0.8 bash run_blanc_naturalqa_test.sh
```

## Code Reference

https://github.com/facebookresearch/SpanBERT

## Copyright

Copyright 2020-present NAVER Corp. and KAIST(Korea Advanced Institute of Science and Technology)

## Acknowledgement

This work was partly supported by NAVER Corp. and Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIP) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding).

