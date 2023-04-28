#!/usr/bin/env python3

import sys
import json

import torch

from argparse import ArgumentParser

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)


def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('prompts')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if torch.cuda.is_available():
        model.to('cuda')
    print('model on', model.device)
    
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )

    with open(args.prompts) as f:
        for l in f:
            data = json.loads(l)
            prompt = data['instruction'] + '\n\n'
            if data['context'] is not None:
                continue    # TODO instructions with context
            output = pipe(
                prompt,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=100
            )
            print(output[0]['generated_text'])
            print('-' * 78)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
