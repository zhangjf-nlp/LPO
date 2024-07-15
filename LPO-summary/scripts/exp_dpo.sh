for preference in human helpful harmless empathetic entertainment
do
  deepspeed train_dpo.py --preference ${preference}
  python test_generate_and_reward_and_ppl.py --dpo_model_path sft/checkpoint/dpo_${preference}_preference/checkpoint
done