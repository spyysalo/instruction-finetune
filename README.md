# Instruction finetuning

Finetune language model on instruction data.

## Example

Get Finnish version of [https://github.com/databrickslabs/dolly/tree/master/data](https://github.com/databrickslabs/dolly/tree/master/data).

```
git clone https://github.com/TurkuNLP/dolly-fi.git
```

Split into a training and validation subset

```
head -n 14000 dolly-fi/dolly-15k-fi.jsonl > train.jsonl
tail -n +14001 dolly-fi/dolly-15k-fi.jsonl > validation.jsonl
```

## Running on LUMI

To run this on the [LUMI supercomputer](https://lumi-supercomputer.eu/):

First, load modules

```
module load cray-python
module load LUMI/22.08 partition/G rocm/5.1.4
```

Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install python modules

```
python -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch==1.12.1+rocm5.1.1 --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
python -m pip install --upgrade transformers datasets evaluate
```

Edit `scripts/gpu-sinteractive.sh` to use the right `--account`

Start interactive session on GPU node

```
./scripts/gpu-sinteractive.sh
```

Load modules and activate venv

```
module load cray-python
module load LUMI/22.08 partition/G rocm/5.1.4
source venv/bin/activate
```

Set cache locations (`$HOME` has limited space)

```
export TRANSFORMERS_CACHE=$PWD/cache
export HF_DATASETS_CACHE=$PWD/cache
export HF_MODULES_CACHE=$PWD/cache
```

Fine-tune

```
python3 train.py TurkuNLP/gpt3-finnish-xl \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    train.jsonl validation.jsonl 
```

Output for running something like the above might initially include e.g.

```
--- Base model ---
Mikä maa on voittanut eniten euroviisuja?

Laajenna klikkaamalla

Ei ole voittanut. Suomi on voittanut kaksi kertaa, vuosina 1961 ja 1974. Vuonna 1961 voitto tuli kappaleella Valoa ikkunassa
```

and after training

```
--- Fine-tuned model ---
Mikä maa on voittanut eniten euroviisuja?

Ruotsi
```

Perhaps unsurpsisingly for a comparatively small model that hasn't
been specifically trained for truthfulness, neither of these outputs
is correct, but the latter is a plausible reponse, whereas the former
is merely plausible as a continuation (specifically on a web page).
