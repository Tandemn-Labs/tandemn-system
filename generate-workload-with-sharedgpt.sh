#!/bin/bash

python3 generate_sharegpt_workload.py \
    --input ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-requests 1000 \
    --avg-input-tokens 10000 \
    --avg-output-tokens 100 \
    --tolerance 0.5 \
    --seed 42
    # --upload-s3