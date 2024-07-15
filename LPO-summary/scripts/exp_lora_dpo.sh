lora_dim=8
for preference in human helpful harmless empathetic entertainment
do
  deepspeed train_dpo.py --preference ${preference} --lora_dim ${lora_dim}
  python test_generate_and_reward_and_ppl.py --lora_model_path sft/checkpoint/lora-${lora_dim}_dpo_${preference}_preference/best_model
done