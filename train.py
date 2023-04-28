#!/usr/bin/python3

# Train causal language model on instruction data

import sys
import os
import json

import torch
import numpy as np

from logging import warning
from argparse import ArgumentParser

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    pipeline,
)


# Avoid "huggingface/tokenizers: The current process just got forked" warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Maximum "reasonable" sequence length
MAX_MAX_LENGTH = 2**16


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--learning-rate', type=float, default=5e-05)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=1)
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--max-train-examples', type=int, default=None)
    ap.add_argument('--max-valid-examples', type=int, default=None)
    ap.add_argument('model')
    ap.add_argument('train_data')
    ap.add_argument('valid_data')
    return ap


def parse_line(line):
    d = json.loads(line)
    if all(k in d for k in ('instruction', 'context', 'response')):
        # Dolly format
        prompt = d['instruction'] + '\n\n'
        if d['context'] and not d['context'].isspace():
            prompt += d['context'] + '\n\n'
        response = d['response']
        return prompt, response
    else:
        # TODO support other formats
        raise ValueError('unrecognized format')


def load_data(fn, max_examples):
    prompts, responses = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            try:
                prompt, response = parse_line(l)
                prompts.append(prompt)
                responses.append(response)
            except Exception as e:
                raise ValueError(f'parsing line {ln} in {fn}: {e}: {l}')
            if max_examples is not None and len(prompts) >= max_examples:
                break
    data = {
        'prompt': prompts,
        'response': responses,
    }
    return Dataset.from_dict(data)


def preprocess(data, tokenizer):
    prompts = data['prompt']
    responses = data['response']
    end_of_prompt = tokenizer.sep_token
    end_of_text = tokenizer.eos_token
    
    combined = []
    for prompt, response in zip(prompts, responses):
        combined.append(prompt + end_of_prompt + response + end_of_text)

    # Truncation would be problematic for this task
    tokenized = tokenizer(combined, truncation=False)

    return tokenized


def get_outputs(ref_ids, pred_ids, tokenizer):
    ref_ids, pred_ids = ref_ids.tolist(), pred_ids.tolist()

    # remove prompts (everything up to the first sep in labels)
    for i in range(len(ref_ids)):
        o = ref_ids[i].index(tokenizer.sep_token_id)
        ref_ids[i] = ref_ids[i][o+1:]
        pred_ids[i] = pred_ids[i][o:]    # labels are shifted + 1

    # remove everything starting at the first remaining sep
    for i in range(len(ref_ids)):
        try:
            o = ref_ids[i].index(tokenizer.sep_token_id)
            ref_ids[i] = ref_ids[i][:o]
        except:
            warning(f'missing sep in refs {i}')

    for i in range(len(pred_ids)):
        try:
            o = pred_ids[i].index(tokenizer.sep_token_id)
            pred_ids[i] = pred_ids[i][:o]
        except:
            pass    # preds don't necessarily have sep

    return ref_ids, pred_ids


def logits_argmax(logits, labels):
    # https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(axis=-1)


class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        data = super().__call__(features, return_tensors)

        end_of_prompt_id = self.tokenizer.sep_token_id
        for i in range(len(data['labels'])):
            eop_indices = np.where(data['labels'][i] == end_of_prompt_id)[0]
            if len(eop_indices) > 0:
                # TODO this should really be eop_indices[0]+1 but that
                # would mask the eop which would mess up the current
                # logic for separating the prompt from the output
                data['labels'][i,:eop_indices[0]] = -100
            else:
                warning('missing eop in labels')

        return data


def get_max_length(model, tokenizer):
    try:
        model_max_length = model.config.max_position_embeddings
        if model_max_length > MAX_MAX_LENGTH:
            warning(f'model.config.max_position_embeddings is '
                    f'{model_max_length}')
    except AttributeError as e:
        warning(f'failed to get max_position_embeddings: {e}')
        model_max_length = 2**64    # something unreasonable

    tokenizer_max_length = tokenizer.model_max_length
    if tokenizer_max_length > MAX_MAX_LENGTH:
        warning(f'tokenizer.model_max_length is {tokenizer_max_length}')

    max_length = min(model_max_length, tokenizer_max_length)

    if max_length > MAX_MAX_LENGTH:
        raise ValueError(f'failed to get max length ({max_length})')
    else:
        return max_length


def filter_by_length(datasetdict, max_length):
    for k in datasetdict:
        dataset = datasetdict[k]
        filtered = dataset.filter(lambda e: len(e['input_ids']) <= max_length)
        orig_length = len(dataset['input_ids'])
        filt_length = len(filtered['input_ids'])
        if filt_length < orig_length:
            warning(
                f'filtered {k} from {orig_length} to {filt_length} '
                f'({filt_length/orig_length:.1%}) by max_length {max_length}'
            )
            datasetdict[k] = filtered

    return datasetdict


def print_generation(label, model, tokenizer, text=None):
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )

    if text is None:
        text = 'Mikä maa on voittanut eniten euroviisuja?\n\n'

    print('---', label, '---')
    print(pipe(text, max_new_tokens=25)[0]['generated_text'])


def main(argv):
    args = argparser().parse_args(argv[1:])

    print('cuda available:', torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    print_generation('Base model', model, tokenizer)
    
    max_length = get_max_length(model, tokenizer)
    print(f'using max_length {max_length}')

    # add special tokens if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    model.resize_token_embeddings(len(tokenizer))

    train_data = load_data(args.train_data, args.max_train_examples)
    valid_data = load_data(args.valid_data, args.max_valid_examples)

    dataset = DatasetDict({
        'train': train_data,
        'validation': valid_data,
    })

    dataset = dataset.map(
        lambda d: preprocess(d, tokenizer),
        batched=True
    )
    dataset = filter_by_length(dataset, max_length)

    for s in ('train', 'validation'):
        print(f'max {s} input_ids length',
              max(len(i) for i in dataset[s]['input_ids']))

    print(
        'Example example:\n'+
        tokenizer.decode(dataset['train']['input_ids'][0]),
    )

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        output_dir='output',
        logging_dir='logs',
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=4,
        #eval_accumulation_steps=1,
        evaluation_strategy='steps',
        logging_strategy='steps',
        weight_decay=0.01,
        num_train_epochs=1,
        eval_steps=1000,
        logging_steps=1000,
        save_strategy='no',
        #save_total_limit=5,
        #save_steps=1000,
        #bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    data_collator = PromptMaskingDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        preprocess_logits_for_metrics=logits_argmax,
    )

    trainer.train()

    valid_results = trainer.evaluate(dataset['validation'])
    print('MODEL:', args.model)
    print('LEARGNING RATE:', args.learning_rate)
    print('BATCH SIZE:', args.batch_size)
    print('GRADIENT ACCUMULATION STEPS:', args.gradient_accumulation_steps)
    print('FINAL VALIDATION LOSS:', valid_results['eval_loss'])

    trainer.save_model('finetuned-model')

    print_generation('Fine-tuned model', model, tokenizer)
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))
