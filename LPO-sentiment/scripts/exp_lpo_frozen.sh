deepspeed train_vae.py --frozen_pretrained 1 --add_skip_connection 1 --output_dir vae_frozen

for preference in positive negative neutral
do
  for seed in 333 666 999
  do
    deepspeed train_lpo.py --preference ${preference} --seed ${seed} --vae_model_path vae_frozen/checkpoint --output_dir vae_frozen/lpo_${preference}_seed_${seed}
    python test_generate_and_reward_and_ppl.py --vae_model_path vae_frozen/checkpoint --lpo_model_path vae_frozen/lpo_${preference}_seed_${seed}/checkpoint
  done
done