# train the P-Tuning SFT model
deepspeed train_sft.py --p_tuning_on_generation --base_model_path sft/checkpoint --learning_rate 1e-3
python test_generate_and_reward_and_ppl.py --pt_model_path pt-sft-on-generation/checkpoint
# ppl:  30.86

# train the P-Tuning DPO model
beta=0.1
# beta  ppl
for preference in Inform Questions Directives Commissive
do
  deepspeed train_dpo.py --beta ${beta} --preference ${preference} --pt_model_path pt-sft-on-generation/checkpoint
  python test_generate_and_reward_and_ppl.py --pt_model_path pt-sft-on-generation/checkpoint/dpo_${preference}_preference_beta_${beta}/checkpoint
done