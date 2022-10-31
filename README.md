# NONE
Code for "Training with More Confidence: Mitigating Injected and Natural Backdoors During Training"

## Environment
requirements.txt

## Prepare Data
- We provide poisonsed datasets:

https://drive.google.com/drive/folders/13oBVDijC-nweBtHw2CBbcevXemc5vTTW. 

- Download and unzip them:
```bash
unzip single_target_p5_s3.zip
```
- You can also generate poisoned datasets by using https://github.com/MadryLab/label-consistent-backdoor-code

## Run Experiments on Defending Injected Trojans
- For example, to run NONE on CIFAR10 against BadNets Attack:

```bash
python -u run_none.py --dataset cifar10 --arch resnet18 --poison_type single_target --none_lr 1e-4 --max_reset_fraction 0.03 --poison_rate 0.05 --epoch_num_1 200 --epoch_num_2 20
```

## Run Experiments on Defending Natural Trojans

- First generate examples for calculating REASR:
```bash
python generate_examples.py --dataset cifar10 --save_dir ./abs/example/
```

- Run NONE to against Natural Trojans on CIFAR10:

```bash
python -u run_none.py --dataset cifar10 --arch nin --poison_type none --none_lr 1e-4 --max_reset_fraction 0.20 --epoch_num_1 200 --epoch_num_2 40 --round_num 10
```

- The results will be recorded in logger file.
