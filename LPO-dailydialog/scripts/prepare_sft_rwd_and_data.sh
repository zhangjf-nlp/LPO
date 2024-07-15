# train the SFT model
deepspeed train_sft.py

# train the reward model
deepspeed train_classifier.py

# test the ground truth with the reward model
python test_generate_and_reward_and_ppl.py --test_ground_truth 1

# test the SFT model with the reward model
python test_generate_and_reward_and_ppl.py --sft_model_path sft/checkpoint

# one2many generation from the SFT model
python dailydialog_one2many_dataset.py

# annotate the intent probs in one2many generation with the reward model
python dailydialog_comparison_dataset.py