# train the P-Tuning SFT model
deepspeed train_sft.py --p_tuning_on_generation --base_model_path sft/checkpoint --mini_batch_size 1 --global_batch_size 8 --epochs 1 --learning_rate 1e-3
python test_generate_and_reward_and_ppl.py --pt_model_path pt-sft-on-gen/checkpoint
# ppl:  30.86

# train the P-Tuning DPO model
beta=0.5
# beta  ppl
# 0.1   43.25
# 0.5   32.88
for preference in positive negative neutral
do
  deepspeed train_dpo.py --beta ${beta} --preference ${preference} --pt_model_path pt-sft-on-gen/checkpoint
  python test_generate_and_reward_and_ppl.py --pt_model_path pt-sft-on-gen/checkpoint/dpo_${preference}_preference_beta_${beta}/checkpoint
done