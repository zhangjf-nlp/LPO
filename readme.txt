We include all our code for model training in this repository, except PPO (we realize in the framework of TRLX, see demo on https://github.com/CarperAI/trlx/tree/main/examples/summarize_rlhf).

Our experiment environment is mainly based on pyTorch + transformer + deepspeed. We include the requirements.txt for the main packages.

To reproduce our results, you can download the corresponding datasets and models according to LPO-xxx/config_and_utils.py and run the sh files in LPO-xxx/scripts.

Unfortunately, the corresponding scripts of LPO-summary is not completely verified yet, which may lead to runtime errors. We will verify and release them as soon as possible.