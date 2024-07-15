#!/bin/bash

working_dir=lpo_exp

deepspeed train_vae.py --add_skip_connection 1 --frozen_pretrained 1 --output_dir ${working_dir}/vae-frozen-base
python test_generate_and_reward_and_ppl.py --vae_model_path ${working_dir}/vae-frozen-base/checkpoint

deepspeed train_vae.py --vae_model_path ${working_dir}/vae-frozen-base/checkpoint --add_contra_loss 1 --frozen_pretrained 0 --output_dir ${working_dir}/vae-unfrozen-base
python test_generate_and_reward_and_ppl.py --vae_model_path ${working_dir}/vae-unfrozen-base/checkpoint

for preference in Inform Questions Directives Commissive
do
  deepspeed train_lpo.py --vae_model_path ${working_dir}/vae-frozen-base/checkpoint --preference ${preference} --output_dir ${working_dir}/lpo-${preference}-frozen-base
  python test_generate_and_reward_and_ppl.py --vae_model_path ${working_dir}/vae-frozen-base/checkpoint --lpo_model_path ${working_dir}/lpo-${preference}-frozen-base/checkpoint
done

for preference in Inform Questions Directives Commissive
do
  deepspeed train_lpo.py --vae_model_path ${working_dir}/vae-unfrozen-base/checkpoint --preference ${preference} --output_dir ${working_dir}/lpo-${preference}-unfrozen-base
  python test_generate_and_reward_and_ppl.py --vae_model_path ${working_dir}/vae-unfrozen-base/checkpoint --lpo_model_path ${working_dir}/lpo-${preference}-unfrozen-base/checkpoint
done