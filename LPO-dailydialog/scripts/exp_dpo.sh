beta=0.1
for preference in Inform Questions Directives Commissive
do
  deepspeed train_dpo.py --preference ${preference}
  python test_generate_and_reward_and_ppl.py --dpo_model_path sft/checkpoint/dpo_${preference}_preference_beta_${beta}/checkpoint
done