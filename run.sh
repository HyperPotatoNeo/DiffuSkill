#!/bin/zsh

source activate cvideos

# python training/train_skills.py --beta 0.1 --conditional_prior 0

python training/train_skills.py --beta 1 --conditional_prior 0

# python training/train_skills.py --beta 0.001 --conditional_prior 0

# python training/train_skills.py --beta 0.1 --conditional_prior 1

# python training/train_skills.py --beta 0.01 --conditional_prior 1

# python training/train_skills.py --beta 0.001 --conditional_prior 1