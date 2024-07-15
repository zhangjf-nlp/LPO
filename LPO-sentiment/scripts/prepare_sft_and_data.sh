# train the SFT model
deepspeed train_sft.py
python test_generate_and_reward_and_ppl.py --sft_model_path sft/checkpoint

# one2many generation from the SFT model
python imdb_one2many_dataset.py

# annotate the sentiment scores in one2many generation
python imdb_comparison_dataset.py