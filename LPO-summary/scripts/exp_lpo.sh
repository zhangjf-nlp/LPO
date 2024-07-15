deepspeed train_vae.py --base_model_path sft/checkpoint --data_from generation --frozen_pretrained 1 --add_contra_loss 1

deepspeed train_vae.py --vae_model_path gptj_vae_frozen_base_contra/checkpoint --data_from openai --frozen_pretrained 0 --add_contra_loss 1

for preference in human helpful harmless empathetic entertainment
do
  deepspeed train_lpo.py --vae_model_path gptj_vae_contra/checkpoint --preference ${preference}
  python test_generate_and_reward_and_ppl.py --vae_model_path gptj_vae_contra/checkpoint --lpo_model_path gptj_vae_contra/checkpoint/lpo_${preference}_preference/checkpoint
done