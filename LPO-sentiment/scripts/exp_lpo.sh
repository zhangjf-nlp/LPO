deepspeed train_vae.py --frozen_pretrained 1 --add_skip_connection 1 --output_dir vae_frozen

deepspeed train_vae.py --frozen_pretrained 0 --add_contra_loss 1 --vae_model_path vae_frozen/checkpoint --output_dir vae_unfrozen

for preference in positive negative neutral
do
  deepspeed train_lpo.py --preference ${preference} --vae_model_path vae_unfrozen/checkpoint --output_dir vae_unfrozen/lpo_${preference}
  python test_generate_and_reward_and_ppl.py --vae_model_path vae_unfrozen/checkpoint --lpo_model_path vae_unfrozen/lpo_${preference}/checkpoint
done