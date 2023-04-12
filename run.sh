#!/bin/zsh

source activate cvideos

# python training/train_skills.py --beta 0.1 --conditional_prior 0

#python training/train_skills.py --beta 1 --conditional_prior 0

# python training/train_skills.py --beta 0.001 --conditional_prior 0

# python training/train_skills.py --beta 0.1 --conditional_prior 1

# python training/train_skills.py --beta 0.01 --conditional_prior 1

# python training/train_skills.py --beta 0.001 --conditional_prior 1

#python3 training/collect_diffusion_data.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_0_diffusion_prior_False_best_a.pth' --horizon 30 --per_element_sigma 1 --conditional_prior 0 --z_dim 64 --append_goals 0 --state_decoder_type autoregressive --save_z_dist 1
#python3 training/collect_diffusion_data.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_0_diffusion_prior_False_best_a.pth' --horizon 30 --per_element_sigma 1 --conditional_prior 0 --z_dim 64 --append_goals 1 --state_decoder_type autoregressive --save_z_dist 1
#python3 training/collect_diffusion_data.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_1_diffusion_prior_False_best_a.pth' --horizon 30 --per_element_sigma 1 --conditional_prior 1 --z_dim 64 --append_goals 0 --state_decoder_type autoregressive --save_z_dist 1
#python3 training/collect_diffusion_data.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_1_diffusion_prior_False_best_a.pth' --horizon 30 --per_element_sigma 1 --conditional_prior 1 --z_dim 64 --append_goals 1 --state_decoder_type autoregressive --save_z_dist 1

#python3 training/train_diffusion.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_0_diffusion_prior_False_best_a.pth' --append_goals 0 --diffusion_steps 200 --sample_z 1 --n_epoch 1000
#python3 training/train_diffusion.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_0_diffusion_prior_False_best_a.pth' --append_goals 1 --diffusion_steps 200 --sample_z 1 --n_epoch 1000
#python3 training/train_diffusion.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_1_diffusion_prior_False_best_a.pth' --append_goals 0 --diffusion_steps 200 --sample_z 1 --n_epoch 1000
#python3 training/train_diffusion.py --skill_model_filename 'skill_model_antmaze-large-diverse-v2_encoderType(gru)_state_dec_autoregressive_policy_dec_autoregressive_H_30_b_1.0_conditionalp_1_diffusion_prior_False_best_a.pth' --append_goals 1 --diffusion_steps 200 --sample_z 1 --n_epoch 1000

python training/train_skills.py --beta 1 --conditional_prior 0 --z_dim 8 --policy_decoder_type mlp
python training/train_skills.py --beta 1 --conditional_prior 1 --z_dim 8 -policy_decoder_type mlp
python training/train_skills.py --beta 1 --conditional_prior 0 --z_dim 16 -policy_decoder_type mlp
python training/train_skills.py --beta 1 --conditional_prior 1 --z_dim 16 -policy_decoder_type mlp