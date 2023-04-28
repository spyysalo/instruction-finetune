#!/bin/bash

srun \
    --account=project_462000241 \
    --partition=small-g \
    --ntasks=1 \
    --gres=gpu:mi250:1 \
    --time=12:00:00 \
    --mem=256G \
    --pty \
    bash
