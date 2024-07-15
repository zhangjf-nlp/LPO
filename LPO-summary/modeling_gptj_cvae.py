import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gptj.modeling_gptj import *

from config_and_utils import BASE_MODEL_PATH


def sampling(mean, logvar, n_samples=1):
    mu = torch.stack([mean] * n_samples, dim=-1)
    sigma = torch.stack([torch.exp(logvar * 0.5)] * n_samples, dim=-1)
    eps = torch.zeros_like(sigma).normal_()
    zs = eps * sigma + mu
    return zs


def log_pdf(mean, logvar, zs):
    import numpy as np
    while len(zs.size()) > len(mean.size()):
        mean = mean.unsqueeze(-1)
        logvar = logvar.unsqueeze(-1)
    return -0.5 * np.log(2 * np.pi) - 0.5 * logvar - \
        (zs - mean).pow(2) / (2 * torch.exp(logvar) + 1e-4)


def dg_kld(config, prior_mean, prior_logvar, post_mean, post_logvar):
    batch_size, dim_z = prior_mean.shape

    n_samples = config.kl_sampling_times
    zs = sampling(post_mean, post_logvar, n_samples)

    priors_mean = prior_mean.unsqueeze(-1).repeat(1, 1, n_samples)
    priors_logvar = prior_logvar.unsqueeze(-1).repeat(1, 1, n_samples)
    # [batch_size, dim_z, n_samples]
    logp_priors_zs = log_pdf(priors_mean, priors_logvar, zs)
    # e.g.
    #         z1 z2 z3
    # prior   .. .. ..
    # [batch_size, dim_z, n_samples]

    zs = zs.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    posts_mean = post_mean.unsqueeze(1).unsqueeze(-1).repeat(1, batch_size, 1, n_samples)
    posts_logvar = post_logvar.unsqueeze(1).unsqueeze(-1).repeat(1, batch_size, 1, n_samples)
    # [batch_size, batch_size, dim_z, n_samples]
    # the first "batch_size" is of prior/post, and the second "batch_size" is of zs
    # in another perspective, the first / second is of aggreagation / stratified sampling
    logp_posts_zs = log_pdf(posts_mean, posts_logvar, zs)
    # e.g.
    #        z1 z2 z3
    # post1  .. .. ..
    # post2  .. .. ..
    # post3  .. .. ..

    if config.marginal_kl:
        # regularization on each dimension respectively
        logp_posts_zs = logp_posts_zs.view(batch_size, batch_size, dim_z, n_samples)
        # [batch_size, batch_size, dim_z, n_samples]
        logp_priors_zs = logp_priors_zs.view(batch_size, dim_z, n_samples)
        # [batch_size, dim_z, n_samples]
    else:
        # regularization in the high-dimensional joint latent space
        logp_posts_zs = logp_posts_zs.sum(dim=-2, keepdims=True)
        # [batch_size, batch_size, 1, n_samples]
        logp_priors_zs = logp_priors_zs.sum(dim=-2, keepdims=True)
        # [batch_size, 1, n_samples]

    # aggregation: post1(z), post2(z), post3(z) -> post_agg(z)

    logp_posts_max = logp_posts_zs.max(dim=0).values
    logp_agg_post_zs = (logp_posts_zs - logp_posts_max).exp().mean(dim=0).log() + logp_posts_max
    # [batch_size, 1 or dim_z, n_samples]
    # e.g. (the dim of post-agg is removed by mean with keepdims=False)
    #           z1 z2 z3
    # post-agg  .. .. ..

    # mote carlo with stratified sampling: post_agg(z1), post_agg(z2), post_agg(z3) -> post_agg(z_agg)

    # mote carlo with stratified sampling: prior(z1), prior(z2), prior(z3) -> prior(z_agg)
    density_gaps = logp_agg_post_zs - logp_priors_zs
    # [batch_size, 1 or dim_z, n_samples]
    kl = density_gaps.mean(dim=0)
    # [1 or dim_z, n_samples]

    kl = kl.sum(dim=0).mean(dim=-1)

    # []
    return kl


class GPTJCVAEConfig(GPTJConfig):
    def __init__(
        self,
        base_pretrained_path=BASE_MODEL_PATH,
        num_q=4,
        dim_z=32,
        num_p=4,
        latent_aggregator_layers=2,
        frozen_pretrained=True,
        use_standard_prior=True,
        marginal_kl=True,
        lm_sampling_times=1,
        kl_sampling_times=16,
        lpo_sampling_times=64,
        without_contra=0,
        without_dg_kld=0,
        add_skip_connection=0,
        add_contra_loss=0,
        beta=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # for model structure
        self.base_pretrained_path = base_pretrained_path
        self.num_q = num_q
        self.dim_z = dim_z
        self.num_p = num_p
        self.latent_aggregator_layers = latent_aggregator_layers
        # for training and inference
        self.frozen_pretrained = frozen_pretrained
        self.use_standard_prior = use_standard_prior
        self.marginal_kl = marginal_kl
        self.lm_sampling_times = lm_sampling_times
        self.kl_sampling_times = kl_sampling_times
        self.lpo_sampling_times = lpo_sampling_times
        self.without_contra = without_contra
        self.without_dg_kld = without_dg_kld
        self.add_skip_connection = add_skip_connection
        self.add_contra_loss = add_contra_loss
        self.beta = beta
        # others
        self.pad_token_id = self.eos_token_id


class LatentAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(torch.Tensor(1, config.num_q, config.n_embd))
        self.queries.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.ln_input = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.blocks = nn.ModuleList([
            GPTJBlock(config)
            for _ in range(config.latent_aggregator_layers)
        ])
        self.ln_output = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_output.bias.data.zero_()
        self.ln_output.weight.data.fill_(1.0)
        self.h2z = nn.Linear(config.n_embd, config.dim_z * 2 // config.num_q)

    def standardize(self):
        # regularize the output latent distribution to standard gaussian
        self.h2z.weight.data.zero_()
        self.h2z.bias.data.zero_()
        return

    def forward(
        self,
        input_embeds: torch.FloatTensor,
        input_embeds_mask: torch.FloatTensor = None,
        fix_entropy: bool = False,
    ):
        dtype, device = self.queries.dtype, self.queries.device
        batch_size, seq_len, hidden_size = input_embeds.shape
        if input_embeds_mask is not None:
            input_embeds_mask = input_embeds_mask.to(dtype=dtype, device=device)  # fp16 compatibility
        else:
            input_embeds_mask = torch.ones([batch_size, seq_len], dtype=dtype, device=device)

        self_attn_mask = torch.ones([batch_size, self.config.num_q], dtype=dtype, device=device)

        attention_mask = torch.cat([input_embeds_mask, self_attn_mask], dim=1)  # [batch_size, to_seq_length]
        attention_mask = attention_mask[:, None, None, :]  # [batch_size, num_heads, from_seq_length, to_seq_length]
        attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min

        hidden_states = torch.cat([input_embeds, self.queries.repeat(batch_size, 1, 1)], dim=1)

        hidden_states = self.ln_input(hidden_states)
        position_ids = torch.arange(seq_len + self.config.num_q, dtype=torch.long, device=device).unsqueeze(0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )[0]
        latent = self.h2z(self.ln_output(hidden_states[:, -self.config.num_q:, :]))
        mean = latent[:, :, :self.config.dim_z // self.config.num_q].reshape(batch_size, self.config.dim_z)
        logvar = latent[:, :, self.config.dim_z // self.config.num_q:].reshape(batch_size, self.config.dim_z)

        if fix_entropy:
            logvar = logvar - logvar.sum(dim=-1, keepdim=True) / self.config.dim_z
            # enforce the entropy of prior distribution equal
            # to that of standard gaussian distribution

        return mean, logvar#, hidden_states


class LatentEncoder(nn.Module):
    def __init__(self, config, fix_entropy=False):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.aggregator = LatentAggregator(config)
        self.fix_entropy = fix_entropy
        self.standardize()

    def load_pretrained(self, pretrained_model):
        self.wte.load_state_dict(pretrained_model.transformer.wte.state_dict())
        for i in range(self.config.latent_aggregator_layers):
            self.aggregator.blocks[i].load_state_dict(pretrained_model.transformer.h[i].state_dict())
        self.aggregator.standardize()

    def standardize(self):
        self.aggregator.standardize()

    def init_wte(self, wte):
        self.wte.load_state_dict(wte.state_dict())

    def init_wte_from_lm_head(self, lm_head):
        self.wte.load_state_dict(lm_head.state_dict(), strict=False)

    def roll_right_padding_to_left(self, right_padded_ids):
        batch_size, seq_len = right_padded_ids.shape
        last_valid_idx = (right_padded_ids != self.config.pad_token_id).sum(dim=1)
        left_padded_ids = right_padded_ids.clone()
        for i in range(batch_size):
            left_padded_ids[i] = torch.roll(left_padded_ids[i], seq_len - last_valid_idx[i].item(), dims=0)
        return left_padded_ids

    def forward(self, input_ids):
        input_ids = self.roll_right_padding_to_left(input_ids).long()
        input_embeds = self.wte(input_ids)
        input_embeds_mask = (input_ids != self.config.eos_token_id).float()
        mean, logvar = self.aggregator(
            input_embeds=input_embeds,
            input_embeds_mask=input_embeds_mask,
            fix_entropy=self.fix_entropy,
        )
        return mean, logvar


class LatentDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z2hs = nn.ModuleList([nn.Sequential(
            nn.Linear(config.dim_z, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd * config.num_p)
        ) for i in range(config.n_layer)])
        #self.z2hs = nn.ModuleList([nn.Linear(config.dim_z, config.n_embd * config.num_p)
        #                           for i in range(config.n_layer)])
        self.lns = nn.ModuleList([nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
                                  for i in range(config.n_layer)])
        self.attns = nn.ModuleList([GPTJAttention(config)
                                    for i in range(config.n_layer)])

    def load_pretrained(self, pretrained_model):
        for i in range(self.config.n_layer):
            self.attns[i].load_state_dict(pretrained_model.transformer.h[i].attn.state_dict())

    def decode(self, zs):
        # input_ps mimic past_key_values in gptj
        # past_key_values: list (in length of n_layers) of past_key and past_value
        # past_key and past_value: (batch_size, num_heads, seq_length, head_features)
        batch_size = zs.shape[0]

        past_key_values = []
        for i in range(self.config.n_layer):
            z2h, ln, attn = self.z2hs[i], self.lns[i], self.attns[i]
            hidden_states = z2h(zs)
            hidden_states = hidden_states.view(batch_size, self.config.num_p, self.config.n_embd)
            hidden_states = ln(hidden_states)
            position_ids = torch.arange(self.config.num_p).unsqueeze(0).repeat(batch_size, 1).to(zs.device)
            present = attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=True
            )[1]
            past_key_values.append(present)

        input_ps = past_key_values
        return input_ps

    def forward(self, mean, logvar):
        # mean, logvar -> zs -> input_ps
        zs = sampling(mean, logvar).squeeze(-1)
        input_ps = self.decode(zs)
        return input_ps


class GPTJForVAE(GPTJPreTrainedModel):
    config_class = GPTJCVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPTJCVAEConfig):
        super().__init__(config)
        self.config = config
        self.latent_encoder = LatentEncoder(config)
        self.latent_decoder = LatentDecoder(config)
        self.base = GPTJForCausalLM(config)
        self.skip_connection = nn.Sequential(
            nn.Linear(self.config.dim_z, self.config.n_embd),
            nn.GELU(),
            nn.Linear(self.config.n_embd, self.config.n_embd)
        )
        self.update_requires_grad_()

    def update_requires_grad_(self):
        self.base.requires_grad_(not self.config.frozen_pretrained)
        self.latent_decoder.attns.requires_grad_(not self.config.frozen_pretrained)

    def load_pretrained(self):
        pretrained_model = GPTJForCausalLM.from_pretrained(self.config.base_pretrained_path)
        self.base.load_state_dict(pretrained_model.state_dict(), strict=False)
        self.latent_encoder.load_pretrained(pretrained_model)
        self.latent_decoder.load_pretrained(pretrained_model)

    def forward(
        self,
        input_ids: torch.LongTensor,  # [1, prompt_len]
        prior_ids: torch.LongTensor,  # [1, input_max_length]
        labels: torch.LongTensor,  # [num_mini, mini_many, output_max_length]
        post_ids: torch.LongTensor,  # [num_mini, mini_many, output_max_length]
        return_loss=True,
        **kwargs,
    ):
        num_mini, mini_many, seq_len = labels.shape
        total_many = mini_many * num_mini
        post_mean, post_logvar = self.latent_encoder(post_ids.view(total_many, post_ids.shape[-1]))
        prior_mean, prior_logvar = torch.zeros_like(post_mean), torch.zeros_like(post_logvar)

        if self.config.without_dg_kld:
            loss_kld = 0.5 * (post_mean.pow(2) + post_logvar.exp() - post_logvar - 1).sum(dim=1).mean(dim=0)
        else:
            loss_kld = dg_kld(self.config, prior_mean, prior_logvar, post_mean, post_logvar)

        # only compute lm loss for the (currently) concerned output
        post_mean, post_logvar = post_mean[:mini_many, :], post_logvar[:mini_many, :]
        input_ps = self.latent_decoder(post_mean, post_logvar)
        # n_layer * [mini_many, num_p, n_embd] <- [mini_many, dim_z]

        prompt_ids = input_ids.repeat(mini_many, 1)
        labels = torch.cat([prompt_ids[:,-1:],labels[0, :, :]], dim=1)
        prompt_ids = prompt_ids[:,:-1]

        prompted_prefix_outputs = self.base.transformer(
            input_ids=prompt_ids,
            past_key_values=input_ps,
            use_cache=True,
            return_dict=True
        )

        if self.config.add_contra_loss or self.config.add_skip_connection:
            unbiased_last_hidden_state = None
            in_batch_nlls = []
            for bias in range(mini_many if self.config.add_contra_loss else 1):
                biased_past_key_values = [
                    (torch.roll(past_keys, shifts=bias, dims=0),
                     torch.roll(past_values, shifts=bias, dims=0))
                    for past_keys, past_values in prompted_prefix_outputs.past_key_values
                ]
                biased_last_hidden_state = self.base.transformer(
                    input_ids=torch.where(labels==-100, self.config.pad_token_id, labels),
                    past_key_values=biased_past_key_values,
                    return_dict=True,
                ).last_hidden_state
                if bias == 0:
                    unbiased_last_hidden_state = biased_last_hidden_state
                    if self.config.add_skip_connection:
                        positive_skip_zs = sampling(post_mean, post_logvar, n_samples=labels.shape[1])
                        skip_hidden_state = F.layer_norm(
                            self.skip_connection(positive_skip_zs.transpose(1, 2)),
                            normalized_shape=[self.config.n_embd],
                            eps=self.config.layer_norm_epsilon
                        )
                        biased_last_hidden_state = biased_last_hidden_state + \
                                                   (skip_hidden_state - skip_hidden_state.detach())
                biased_lm_logits = self.base.lm_head(biased_last_hidden_state)
                nlls = F.cross_entropy(
                    input=biased_lm_logits[:, :-1, :].contiguous().transpose(1, 2),
                    target=labels[:, 1:].contiguous(),
                    reduction='none'
                ).sum(dim=-1)
                in_batch_nlls.append(nlls)

            if self.config.add_contra_loss:
                assert len(in_batch_nlls) == mini_many
                in_batch_nlls = torch.stack(in_batch_nlls, dim=0)
                loss_contra = F.cross_entropy(
                    input=-in_batch_nlls.T,
                    target=torch.zeros(mini_many, dtype=torch.int64, device=in_batch_nlls.device),
                    reduction='mean'
                )
                lm_logits = self.base.lm_head(unbiased_last_hidden_state)
                loss_lm = F.cross_entropy(
                    input=lm_logits[:, :-1, :].contiguous().transpose(1, 2),
                    target=labels[:, 1:].contiguous(),
                    reduction='none'
                ).sum(dim=-1).mean()
            else:
                assert self.config.add_skip_connection
                loss_contra = torch.zeros_like(input_ps[0][0].sum())
                loss_lm = in_batch_nlls[0].mean()
        else:
            loss_contra = torch.zeros_like(input_ps[0][0].sum())
            unbiased_last_hidden_state = self.base.transformer(
                input_ids=torch.where(labels==-100, self.config.pad_token_id, labels),
                past_key_values=prompted_prefix_outputs.past_key_values,
                return_dict=True,
            ).last_hidden_state
            lm_logits = self.base.lm_head(unbiased_last_hidden_state)
            loss_lm = F.cross_entropy(
                input=lm_logits[:, :-1, :].contiguous().transpose(1, 2),
                target=labels[:, 1:].contiguous(),
                reduction='none'
            ).sum(dim=-1).mean()

        loss_vae = loss_lm + loss_kld + loss_contra

        return loss_vae, loss_lm, loss_kld, loss_contra

    def load_lpo_policy_latent_encoder(self, path_to_lpo_model):
        from config_and_utils import load_checkpoint
        lpo_sd = load_checkpoint(path_to_lpo_model, device=self.latent_encoder.aggregator.queries.device)
        self.latent_encoder.load_state_dict({
            k[len("policy_latent_encoder."):]:v for k,v in lpo_sd.items()
            if k.startswith("policy_latent_encoder.")
        })
        self.config.use_standard_prior = False

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        streamer=None,
        post_labels: torch.LongTensor=None,
        post_latent: Tuple[torch.FloatTensor]=None,
        standard_prior: bool=False,
        latent_sampling: bool=False,
        **kwargs,
    ):
        input_ids = kwargs["input_ids"]

        if post_latent is not None:
            mean, logvar = post_latent
        elif post_labels is not None:
            mean, logvar = self.latent_encoder(post_labels)
        elif standard_prior:
            mean, logvar = self.latent_encoder(input_ids)
            mean, logvar = torch.zeros_like(mean), torch.zeros_like(logvar)
        else:
            mean, logvar = self.latent_encoder(input_ids)

        if not latent_sampling:
            logvar.fill_(-100)

        input_ps = self.latent_decoder(mean, logvar)

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids[:, :-1]!=self.config.pad_token_id).long()
        ], dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, self.config.num_p:]
        outputs = self.base.transformer(
            input_ids=input_ids[:, :-1],
            past_key_values=input_ps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        kwargs["past_key_values"] = outputs.past_key_values

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids!=self.config.pad_token_id).long()
        ], dim=1)
        kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = True

        return self.base.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **kwargs,
        )


class GPTJForCVAE(GPTJPreTrainedModel):
    config_class = GPTJCVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPTJCVAEConfig):
        super().__init__(config)
        self.config = config
        self.base = GPTJForCausalLM(config)
        if config.frozen_pretrained:
            self.base.requires_grad_(False)
        self.prior_latent_encoder = LatentEncoder(config)
        self.post_latent_encoder = LatentEncoder(config)
        self.latent_decoder = LatentDecoder(config)

    def load_pretrained(self):
        pretrained_model = GPTJForCausalLM.from_pretrained(self.config.base_model_path)
        self.base.load_state_dict(pretrained_model.state_dict())
        if self.config.frozen_pretrained:
            self.base.requires_grad_(False)
        self.prior_latent_encoder.init_wte(self.base.transformer.wte)
        self.post_latent_encoder.init_wte(self.base.transformer.wte)
        self.prior_latent_encoder.standardize()
        self.post_latent_encoder.standardize()

    def load_lpo_policy_latent_encoder(self, path_to_lpo_model):
        lpo_model = LPOModel(self.config)
        if os.path.exists(os.path.join(path_to_lpo_model, "pytorch_model.bin")):
            lpo_model.load_state_dict(torch.load(os.path.join(path_to_lpo_model, "pytorch_model.bin")))
        elif os.path.exists(os.path.join(path_to_lpo_model, "model.safetensors")):
            from safetensors.torch import load_file
            lpo_model.load_state_dict(load_file(os.path.join(path_to_lpo_model, "model.safetensors")))
        else:
            raise Exception(f"checkpoint file not found in {os.listdir(path_to_lpo_model)}")
        self.prior_latent_encoder.load_state_dict(lpo_model.policy_latent_encoder.state_dict())
        self.config.use_standard_prior = False

    def load_pt_latent_decoder(self, pt_model_path):
        pt_model = PTModel(config=self.config)
        pt_model.load_state_dict(torch.load(os.path.join(pt_model_path, "pytorch_model.bin")))
        for z2k in pt_model.latent_decoder.z2ks:
            z2k.weight.data /= z2k.weight.data.norm(dim=-1).unsqueeze(-1)/self.config.latent_weight_init_norm
        for z2v in pt_model.latent_decoder.z2vs:
            z2v.weight.data /= z2v.weight.data.norm(dim=-1).unsqueeze(-1)/self.config.latent_weight_init_norm
        self.latent_decoder.load_state_dict(pt_model.latent_decoder.state_dict())

    def load_vae_state_dict(self, vae_model_path):
        gptj_vae = GPTJForVAE.from_pretrained(vae_model_path)
        self.post_latent_encoder.load_state_dict(gptj_vae.latent_encoder.state_dict())
        self.latent_decoder.load_state_dict(gptj_vae.latent_decoder.state_dict(), strict=False)
        self.base.load_state_dict(gptj_vae.base.state_dict(), strict=False)

    # for training & evaluation only
    def forward(
        self,
        input_ids: torch.LongTensor,  # [1, prompt_len]
        labels: torch.LongTensor,  # [num_mini, mini_many, seq_len]
        prior_ids: torch.LongTensor,  # [1, seq_len]
        post_ids: torch.LongTensor,  # [num_mini, mini_many, ans_seq_len]
        return_loss=True,
        **kwargs,
    ):
        num_mini, mini_many, seq_len = labels.shape
        total_many = mini_many * num_mini
        post_mean, post_logvar = self.post_latent_encoder(post_ids.view(total_many, post_ids.shape[-1]))
        if self.config.use_standard_prior:
            prior_mean, prior_logvar = torch.zeros_like(post_mean), torch.zeros_like(post_logvar)
        else:
            prior_mean, prior_logvar = self.prior_latent_encoder(prior_ids)
            prior_mean, prior_logvar = prior_mean.repeat(total_many, 1), prior_logvar.repeat(total_many, 1)

        # only compute lm loss for the (currently) concerned output
        input_ps = self.latent_decoder(post_mean[:mini_many, :], post_logvar[:mini_many, :])
        # n_layer * [mini_many, num_p, n_embd] <- [mini_many, dim_z]
        labels = labels[0, :, :]
        # [mini_many, seq_len]
        input_ids = input_ids.repeat(mini_many, 1)
        # [mini_many, prompt_len]

        outputs = self.base.transformer(
            input_ids=input_ids,
            past_key_values=input_ps,
            use_cache=True,
            return_dict=True
        )
        past_key_values = outputs.past_key_values
        last_hidden_state = outputs.last_hidden_state

        in_batch_nlls = torch.stack([
            torch.roll(
                self.get_conditional_nlls(
                    input_ids,
                    last_hidden_state,
                    past_key_values,
                    torch.roll(labels, shifts=i, dims=0)
                ), shifts=-i, dims=0
            ) for i in range(1 if self.config.without_contra else mini_many)
        ], dim=0)
        # -log p(y_j|x,z_(i+j))
        # e.g.
        # [[(y_0|x,z_0), (y_1|x,z_1), (y_2|x,z_2)], -> positive
        #  [(y_0|x,z_1), (y_1|x,z_2), (y_2|x,z_0)], -> negative
        #  [(y_0|x,z_2), (y_1|x,z_0), (y_2|x,z_1)]] -> negative

        nlls_positive = in_batch_nlls[0]
        if self.config.without_contra:
            loss_contrastive = torch.zeros_like(nlls_positive.mean())
        else:
            loss_contrastive = torch.nn.functional.cross_entropy(
                input=-in_batch_nlls.T,
                target=torch.zeros(mini_many, dtype=torch.int64, device=in_batch_nlls.device)
            ).mean()

        loss_kld = dg_kld(self.config, prior_mean, prior_logvar, post_mean, post_logvar)
        if self.config.without_dg_kld:
            standard_kld = 0.5 * (post_mean.pow(2) + post_logvar.exp() - post_logvar - 1).sum(dim=1).mean(dim=0)
            loss_cvae = nlls_positive.mean() + standard_kld
        else:
            loss_cvae = nlls_positive.mean() + loss_kld

        loss = loss_cvae + loss_contrastive

        return loss, loss_cvae, loss_contrastive, loss_kld

    def get_conditional_nlls(self, input_ids, input_ids_hidden_state, past_key_values, labels):
        labels_hidden_state = self.base.transformer(
            input_ids=torch.where(labels == -100, self.config.pad_token_id, labels),
            past_key_values=past_key_values,
            return_dict=True,
        ).last_hidden_state
        hidden_state = torch.cat([input_ids_hidden_state, labels_hidden_state], dim=1)
        lm_logits = self.base.lm_head(hidden_state)
        nlls = F.cross_entropy(
            input=lm_logits[:, :-1, :].contiguous().transpose(1, 2),
            target=torch.cat([input_ids.clone().fill_(-100), labels], dim=1)[:, 1:].contiguous(),
            reduction='none'
        ).sum(dim=-1)
        return nlls

    def posterior_evaluation(
        self,
        input_ids: torch.LongTensor,  # [1, prompt_len]
        labels: torch.LongTensor,  # [num_mini, mini_many, seq_len]
        prior_ids: torch.LongTensor,  # [1, seq_len]
        post_ids: torch.LongTensor,  # [num_mini, mini_many, ans_seq_len])
        n_samples = 32
    ):
        num_mini, mini_many, seq_len = labels.shape
        total_many = mini_many * num_mini
        post_mean, post_logvar = self.post_latent_encoder(post_ids.view(total_many, post_ids.shape[-1]))
        if self.config.use_standard_prior:
            prior_mean, prior_logvar = torch.zeros_like(post_mean), torch.zeros_like(post_logvar)
        else:
            prior_mean, prior_logvar = self.prior_latent_encoder(prior_ids)
            prior_mean, prior_logvar = prior_mean.repeat(total_many, 1), prior_logvar.repeat(total_many, 1)

        # now we begin to calculate KL(q(z|x,y)||p(z|x,y))
        # where p(z|x,y) = p(y|x,z)p(z|x)/p(y|x)

        # step1. sampling zs
        zs_post = sampling(post_mean, post_logvar, n_samples=n_samples)
        # [total_many, dim_z, n_samples]
        zs_prior = sampling(prior_mean, prior_logvar, n_samples=n_samples)
        # [total_many, dim_z, n_samples]
        zs = torch.cat([zs_post, zs_prior], dim=-1)
        # [total_many, dim_z, n_samples * 2]

        # step2. log q(z|x,y)
        log_qz_y = log_pdf(post_mean, post_logvar, zs).sum(dim=1)

        # step3. log p(z|x,y)
        # step3.1 log p(y|x,z)
        log_py_z = []
        input_ids = input_ids.repeat(total_many, 1)
        labels = labels.view(total_many, labels.shape[-1])
        for iz in range(zs.shape[-1]):
            outputs = self.base.transformer(
                input_ids=input_ids,
                past_key_values=self.latent_decoder.decode(zs[:,:,iz]),
                use_cache=True,
                return_dict=True
            )
            log_py_zi = -self.get_conditional_nlls(
                input_ids,
                outputs.last_hidden_state,
                outputs.past_key_values,
                labels
            )
            log_py_z.append(log_py_zi)
        log_py_z = torch.stack(log_py_z, dim=-1)
        # step3.2 log p(z|x)
        log_pz = log_pdf(prior_mean, prior_logvar, zs).sum(dim=1)
        # step3.3 log p(y|x)
        #log_py = log_py_z.exp().mean(dim=-1, keepdims=True).log()
        bias = log_py_z.max(dim=1, keepdims=True).values
        log_py = (log_py_z-bias).exp().mean(dim=-1, keepdims=True).log() + bias
        # step3.4 
        log_pz_y = log_py_z + log_pz - log_py

        # step4. monte carlo estimation

        # importance_weight = log_qz_y.exp() * 2 / (log_qz_y.exp() + log_pz.exp())
        # importance_weight = (log_qz_y-bias).exp() * 2 / ((log_qz_y-bias).exp() + (log_pz-bias).exp())
        bias = torch.where(log_qz_y>log_pz, log_qz_y, log_pz)
        qz_y_ = (log_qz_y-bias).exp()
        pz_ = (log_pz-bias).exp()
        importance_weight = qz_y_ * 2 / (qz_y_ + pz_)

        kld = ((log_qz_y - log_pz_y) * importance_weight).mean(dim=-1)

        return kld

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        streamer=None,
        post_labels: torch.LongTensor=None,
        post_latent: Tuple[torch.FloatTensor]=None,
        standard_prior: bool=False,
        latent_sampling: bool=False,
        **kwargs,
    ):
        input_ids = kwargs["input_ids"]

        if post_latent is not None:
            mean, logvar = post_latent
        elif post_labels is not None:
            mean, logvar = self.post_latent_encoder(post_labels)
        elif standard_prior:
            mean, logvar = self.prior_latent_encoder(input_ids)
            mean, logvar = torch.zeros_like(mean), torch.zeros_like(logvar)
        else:
            mean, logvar = self.prior_latent_encoder(input_ids)

        if not latent_sampling:
            logvar.fill_(-100)

        input_ps = self.latent_decoder(mean, logvar)

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids[:, :-1]!=self.config.pad_token_id).long()
        ], dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, self.config.num_p:]
        outputs = self.base.transformer(
            input_ids=input_ids[:, :-1],
            past_key_values=input_ps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        kwargs["past_key_values"] = outputs.past_key_values

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids!=self.config.pad_token_id).long()
        ], dim=1)
        kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = True

        return self.base.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **kwargs,
        )


class LPOModel(nn.Module):
    main_input_name = "prompt_ids"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prior_latent_encoder = LatentEncoder(config)
        self.prior_latent_encoder.requires_grad_(False)
        self.post_latent_encoder = LatentEncoder(config)
        self.post_latent_encoder.requires_grad_(False)
        self.policy_latent_encoder = LatentEncoder(config)

    @classmethod
    def from_cvae(cls, cvae_model_path, **kwargs):
        gptj_cvae = GPTJForCVAE.from_pretrained(cvae_model_path)
        config = gptj_cvae.config
        config.update(kwargs)
        model = cls(config)
        model.prior_latent_encoder.load_state_dict(gptj_cvae.prior_latent_encoder.state_dict())
        model.policy_latent_encoder.load_state_dict(gptj_cvae.prior_latent_encoder.state_dict())
        model.post_latent_encoder.load_state_dict(gptj_cvae.post_latent_encoder.state_dict())
        return model

    @classmethod
    def from_vae(cls, vae_model_path, **kwargs):
        gptj_vae = GPTJForVAE.from_pretrained(vae_model_path)
        config = gptj_vae.config
        config.update(kwargs)
        model = cls(config)
        model.policy_latent_encoder.load_state_dict(gptj_vae.latent_encoder.state_dict())
        model.policy_latent_encoder.standardize()
        model.post_latent_encoder.load_state_dict(gptj_vae.latent_encoder.state_dict())
        return model

    def forward(
        self,
        prompt_ids: torch.LongTensor, # x
        chosen_ids: torch.LongTensor, # y_w
        rejected_ids: torch.LongTensor, # y_l
    ):
        prior_ids, chosen_post_ids, rejected_post_ids = prompt_ids, chosen_ids, rejected_ids
        batch_size = prior_ids.shape[0]

        # pi_theta(z|x)
        policy_mean, policy_logvar = self.policy_latent_encoder(prior_ids)
        with torch.no_grad():
            # pi_ref(z|x)
            if self.config.use_standard_prior:
                prior_mean, prior_logvar = torch.zeros_like(policy_mean), torch.zeros_like(policy_logvar)
            else:
                prior_mean, prior_logvar = self.prior_latent_encoder(prior_ids)
            # q(z|x,y_w), and q(z|x,y_l)
            chosen_post_mean, chosen_post_logvar = self.post_latent_encoder(chosen_post_ids)
            rejected_post_mean, rejected_post_logvar = self.post_latent_encoder(rejected_post_ids)
        # batch_size, dim_z

        n_samples = self.config.lpo_sampling_times
        prior_zs = sampling(prior_mean, prior_logvar, n_samples=n_samples)
        chosen_zs = sampling(chosen_post_mean, chosen_post_logvar, n_samples=n_samples)
        rejected_zs = sampling(rejected_post_mean, rejected_post_logvar, n_samples=n_samples)
        zs = torch.cat([prior_zs, chosen_zs, rejected_zs], dim=-1)
        n_samples = zs.shape[-1]
        # batch_size, dim_z, n_samples

        # compare different zs according to posterior likelihoods
        with torch.no_grad():
            # -log q(z|x,y_w)
            zs_nlls_given_chosen = -log_pdf(chosen_post_mean, chosen_post_logvar, zs).sum(dim=1)
            # -log q(z|x,y_l)
            zs_nlls_given_rejected = -log_pdf(rejected_post_mean, rejected_post_logvar, zs).sum(dim=1)

            # log q(z|x,y_w)/q(z|x,y_l), greater is better
            score_zs = zs_nlls_given_rejected - zs_nlls_given_chosen
            # batch_size, n_samples

            # compare zs in pairs
            comparison_matrix = score_zs[:, :, None] - score_zs[:, None, :]
            comparison_matrix = comparison_matrix.detach()
            # batch_size, n_samples, n_samples

        # perform DPO on zs according to the comparison results
        zs_refer_nlls = -log_pdf(prior_mean, prior_logvar, zs).sum(dim=1)
        zs_policy_nlls = -log_pdf(policy_mean, policy_logvar, zs).sum(dim=1)
        zs_policy_reward = zs_refer_nlls.detach() - zs_policy_nlls
        # batch_size, n_samples

        zs_policy_reward_left = zs_policy_reward[:, :, None].repeat(1, 1, n_samples)
        zs_policy_reward_right = zs_policy_reward[:, None, :].repeat(1, n_samples, 1)
        # batch_size, n_samples, n_samples

        left_is_chosen = comparison_matrix > 0
        reward_chosen = torch.where(left_is_chosen, zs_policy_reward_left, zs_policy_reward_right)
        reward_rejected = torch.where(left_is_chosen, zs_policy_reward_right, zs_policy_reward_left)
        # batch_size, n_samples, n_samples

        loss = -F.logsigmoid((reward_chosen - reward_rejected) * self.config.beta)
        acc = (reward_chosen > reward_rejected).float()

        # mask self v.s. self
        comparison_mask = 1 - torch.eye(n_samples).float()
        comparison_mask = comparison_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        comparison_mask = comparison_mask.to(comparison_matrix.device)
        # batch_size, n_samples, n_samples
        loss = (loss * comparison_mask).sum() / comparison_mask.sum()
        acc = (acc * comparison_mask).sum() / comparison_mask.sum()

        return loss, acc


class DPOModel(GPTJPreTrainedModel):
    config_class = GPTJConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config):
        super().__init__(config)
        self.base = GPTJForCausalLM(config)
        self.beta = config.beta

    def load_pretrained(self, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = GPTJForCausalLM.from_pretrained(self.config.base_model_path)
        self.base.load_state_dict(pretrained_model.state_dict())

    def prepare_refer_model(self):
        self.refer_model = copy.deepcopy(self)
        self.refer_model.requires_grad_(False)
        self.refer_model.eval()

    def compute_nlls(self, input_ids, labels):
        hidden_states = self.base.transformer(input_ids=input_ids)[0]
        lm_logits = self.base.lm_head(hidden_states).to(torch.float32)
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        nlls = F.cross_entropy(
            input=shift_logits.transpose(1, 2),
            target=shift_labels,
            reduction='none'
        ).sum(dim=-1)
        return nlls

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        refer_nlls: torch.FloatTensor = None,
        reward: torch.FloatTensor = None,  # r(x,y)
    ):
        theta_nlls = self.compute_nlls(input_ids=input_ids, labels=labels)
        if refer_nlls is None:
            assert hasattr(self, "refer_model")
            with torch.no_grad():
                refer_nlls = self.refer_model.compute_nlls(input_ids=input_ids, labels=labels)
        reward = refer_nlls.detach() - theta_nlls
        chosen_reward = reward[::2]
        rejected_reward = reward[1::2]
        loss = -F.logsigmoid((chosen_reward - rejected_reward) * self.beta)
        acc = (chosen_reward > rejected_reward).float()
        return loss.mean(), acc.mean()


class PTModel(GPTJPreTrainedModel):
    config_class = GPTJCVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPTJCVAEConfig):
        super().__init__(config)
        self.base = GPTJForCausalLM(config)
        self.base.requires_grad_(False)
        self.hs = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(1, config.num_q, config.n_embd))
            for i in range(config.n_layer)
        ])
        for param in self.hs:
            param.data.normal_(mean=0.0, std=config.initializer_range)
        self.lns = nn.ModuleList([nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
                                  for i in range(config.n_layer)])
        self.attns = nn.ModuleList([GPTJAttention(config)
                                    for i in range(config.n_layer)])
        self.attns.requires_grad_(False)
        self.beta = config.beta
        self.mode = "sft"

    def load_pretrained(self, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = GPTJForCausalLM.from_pretrained(self.config.base_model_path)
        self.base.load_state_dict(pretrained_model.state_dict())
        for i in range(self.config.n_layer):
            self.attns[i].load_state_dict(pretrained_model.transformer.h[i].attn.state_dict())

    def switch_into_dpo_mode(self):
        self.mode = "dpo"
        self.refer_model = copy.deepcopy(self)
        self.refer_model.requires_grad_(False)

    def get_input_ps(self, input_ids):
        batch_size = input_ids.shape[0]
        past_key_values = []
        for i in range(self.config.n_layer):
            hidden_states, ln, attn = self.hs[i], self.lns[i], self.attns[i]
            hidden_states = hidden_states.repeat(batch_size, 1, 1)
            hidden_states = ln(hidden_states)
            position_ids = torch.arange(self.config.num_p).unsqueeze(0).repeat(batch_size, 1).to(hidden_states.device)
            present = attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=True
            )[1]
            past_key_values.append(present)
        input_ps = past_key_values
        return input_ps

    def compute_nlls(self, input_ids, labels):
        input_ps = self.get_input_ps(input_ids)
        hidden_states = self.base.transformer.forward(
            input_ids=input_ids,
            past_key_values=input_ps
        )[0]
        lm_logits = self.base.lm_head(hidden_states).to(torch.float32)
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        nlls = F.cross_entropy(
            input=shift_logits.transpose(1, 2),
            target=shift_labels,
            reduction='none'
        ).sum(dim=-1)
        return nlls

    def forward(self, input_ids, labels, refer_nlls=None):
        if self.mode == "sft":
            # supervised fine-tuning
            theta_nlls = self.compute_nlls(input_ids=input_ids, labels=labels)
            return {"loss": theta_nlls.mean()}
        elif self.mode == "dpo":
            # dpo
            theta_nlls = self.compute_nlls(input_ids=input_ids, labels=labels)
            if refer_nlls is None:
                assert hasattr(self, "refer_model")
                self.refer_model.eval()
                with torch.no_grad():
                    refer_nlls = self.refer_model.compute_nlls(input_ids=input_ids, labels=labels)
            log_iw = refer_nlls.detach() - theta_nlls
            chosen_log_iw = log_iw[::2]
            rejected_log_iw = log_iw[1::2]
            disadvantage = (rejected_log_iw - chosen_log_iw) * self.beta
            loss = -F.logsigmoid(-disadvantage)
            acc = (loss < 0.6931).float()
            return loss.mean(), acc.mean()
        else:
            raise NotImplementedError(f"mode: {self.mode}")

    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        streamer=None,
        **kwargs,
    ):
        input_ids = kwargs["input_ids"]
        input_ps = self.get_input_ps(input_ids)

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids[:, :-1]!=self.config.pad_token_id).long()
        ], dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, self.config.num_p:]
        outputs = self.base.transformer(
            input_ids=input_ids[:, :-1],
            past_key_values=input_ps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True
        )
        kwargs["past_key_values"] = outputs.past_key_values

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids!=self.config.pad_token_id).long()
        ], dim=1)
        kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = True

        return self.base.generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **kwargs,
        )

