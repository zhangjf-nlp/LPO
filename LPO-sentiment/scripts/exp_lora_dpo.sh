beta=0.1
lora_dim=8
for preference in positive negative neutral
do
  deepspeed train_dpo.py --preference ${preference} --lora_dim ${lora_dim} --learning_rate 1e-5
  python test_generate_and_reward_and_ppl.py --lora_model_path sft/checkpoint/lora-${lora_dim}_dpo_${preference}_preference_beta_${beta}/best_model
done