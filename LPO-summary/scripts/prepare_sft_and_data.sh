# train the SFT model
deepspeed train_sft.py
python test_generate_and_reward_and_ppl.py --sft_model_path sft/checkpoint

# one2many generation from the SFT model
