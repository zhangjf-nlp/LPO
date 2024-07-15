import copy

import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import *

from config_and_utils import BASE_MODEL_PATH, load_checkpoint


def sampling(mean, logvar, n_samples=1):
    mu = torch.stack([mean] * n_samples, dim=-1)
    sigma = torch.stack([torch.exp(logvar * 0.5)] * n_samples, dim=-1)
    eps = torch.zeros_like(sigma).normal_()
    zs = eps * sigma + mu
    return zs


def log_pdf(mean, logvar, zs) -> torch.Tensor:
    import numpy as np
    if len(zs.shape) == len(mean.shape) + 1:
        mean = mean.unsqueeze(-1)
        logvar = logvar.unsqueeze(-1)
    return -0.5 * np.log(2 * np.pi) - 0.5 * logvar - \
           (zs - mean).pow(2) / (2 * torch.exp(logvar) + 1e-4)


def dgkld(config, prior_mean, prior_logvar, post_mean, post_logvar):
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


class GPT2CVAEConfig(GPT2Config):
    def __init__(
        self,
        base_model_path=BASE_MODEL_PATH,
        num_q=4,
        dim_z=32,
        num_p=4,
        latent_aggregator_layers=2,
        post_with_x=0,
        frozen_pretrained=True,
        use_standard_prior=True,
        marginal_kl=True,
        lm_sampling_times=1,
        kl_sampling_times=16,
        lpo_sampling_times=64,
        latent_weight_init_norm=0.1,
        without_contra=0,
        without_dg_kld=0,
        with_bn=0,
        contra_loss_weight=1.0,
        two_stage_contra=0,
        add_skip_connection=0,
        add_skip_residue=0,
        add_contra_loss=0,
        add_skip_residue_contra=0,
        lpo_post_sampling=0,
        lpo_evidence_ratio=0.5,
        marginal_lpo=0,
        expectation_lpo=0,
        beta=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # for model structure
        self.base_model_path = base_model_path
        self.num_q = num_q
        self.dim_z = dim_z
        self.num_p = num_p
        self.latent_aggregator_layers = latent_aggregator_layers
        self.post_with_x = post_with_x
        # for training and inference
        self.frozen_pretrained = frozen_pretrained
        self.use_standard_prior = use_standard_prior
        self.marginal_kl = marginal_kl
        self.lm_sampling_times = lm_sampling_times
        self.kl_sampling_times = kl_sampling_times
        self.lpo_sampling_times = lpo_sampling_times
        self.latent_weight_init_norm = latent_weight_init_norm
        self.without_contra = without_contra
        self.without_dg_kld = without_dg_kld
        self.with_bn = with_bn
        self.contra_loss_weight = contra_loss_weight
        self.two_stage_contra = two_stage_contra
        self.add_skip_connection = add_skip_connection
        self.add_skip_residue = add_skip_residue
        self.add_contra_loss = add_contra_loss
        self.add_skip_residue_contra = add_skip_residue_contra
        self.lpo_post_sampling = lpo_post_sampling
        self.lpo_evidence_ratio = lpo_evidence_ratio
        self.marginal_lpo = marginal_lpo
        self.expectation_lpo = expectation_lpo
        self.beta = beta
        # others
        self.pad_token_id = self.eos_token_id


class LatentAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.queries = nn.Parameter(torch.Tensor(1, config.num_q, config.n_embd))
        self.queries.data.normal_(mean=0.0, std=config.initializer_range)
        self.ln_input = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.blocks = nn.ModuleList([
            GPT2Block(config)
            for _ in range(config.latent_aggregator_layers)
        ])
        self.ln_output = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
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

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
            )[0]
        latent = self.h2z(self.ln_output(hidden_states[:, -self.config.num_q:, :]))
        mean = latent[:, :, :latent.shape[-1]//2].reshape(batch_size, self.config.dim_z)
        logvar = latent[:, :, latent.shape[-1]//2:].reshape(batch_size, self.config.dim_z)

        if fix_entropy:
            logvar = logvar - logvar.sum(dim=-1, keepdim=True) / self.config.dim_z
            # enforce the entropy of prior distribution equal
            # to that of standard gaussian distribution

        return mean, logvar


class LatentEncoder(nn.Module):
    def __init__(self, config, fix_entropy=False):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.aggregator = LatentAggregator(config)
        self.fix_entropy = fix_entropy
        self.aggregator.standardize()

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
        input_embeds_mask = (input_ids != self.config.pad_token_id).float()
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
        """
        self.z2hs = nn.ModuleList([nn.Linear(config.dim_z, config.n_embd * config.num_p)
                                  for i in range(config.n_layer)])
        """
        self.lns = nn.ModuleList([nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
                                  for i in range(config.n_layer)])
        self.attns = nn.ModuleList([GPT2Attention(config, layer_idx=i)
                                    for i in range(config.n_layer)])
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

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
            present = attn(
                hidden_states=hidden_states,
                use_cache=True,
            )[1]
            past_key_values.append(present)
        
        input_ps = past_key_values
        return input_ps

    def forward(self, mean, logvar):
        # mean, logvar -> zs -> input_ps
        if self.config.lm_sampling_times == 0:
            zs = mean
        else:
            zs = sampling(mean, logvar).squeeze(-1)
        input_ps = self.decode(zs)
        return input_ps

    def get_latent_nlls(self, zs, mean, logvar):
        return -log_pdf(mean, logvar, zs).sum(dim=-1)


class GPT2ForVAE(GPT2PreTrainedModel):
    config_class = GPT2CVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPT2CVAEConfig):
        super().__init__(config)
        self.config = config
        self.latent_encoder = LatentEncoder(config)
        self.latent_decoder = LatentDecoder(config)
        self.base = GPT2LMHeadModel(config)
        self.skip_connection = nn.Sequential(
            nn.Linear(self.config.dim_z, self.config.n_embd),
            nn.GELU(),
            nn.Linear(self.config.n_embd, self.config.n_embd)
        )

        if self.config.with_bn:
            self.bn = nn.BatchNorm1d(self.config.dim_z)
            self.bn.weight.requires_grad = False
            self.bn.weight.data.fill_(0.5)

        self.update_requires_grad_()

    def update_requires_grad_(self):
        self.base.requires_grad_(not self.config.frozen_pretrained)
        self.latent_decoder.attns.requires_grad_(not self.config.frozen_pretrained)

    def load_pretrained(self, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = GPT2LMHeadModel.from_pretrained(self.config.base_model_path)
        self.base.load_state_dict(pretrained_model.state_dict(), strict=False)
        self.latent_encoder.load_pretrained(pretrained_model)
        self.latent_decoder.load_pretrained(pretrained_model)

    def skip_grad_lm_loss(self, post_mean, post_logvar, input_ps, input_ids, labels, grad_weight=1.0, reduction="mean"):
        last_hidden_state = self.base.transformer(
            input_ids=input_ids,
            past_key_values=input_ps,
            return_dict=True,
        ).last_hidden_state
        if self.config.lm_sampling_times == 0:
            zs = post_mean.unsqueeze(-1).repeat(1, 1, last_hidden_state.shape[1])
        else:
            zs = sampling(post_mean, post_logvar, n_samples=last_hidden_state.shape[1])
        # [batch_size, dim_z, seq_len]
        skip_hidden_state = F.layer_norm(self.skip_connection(zs.transpose(1, 2)),
                                         normalized_shape=[self.config.n_embd],
                                         eps=self.config.layer_norm_epsilon)
        skip_residue = (skip_hidden_state - skip_hidden_state.detach()) \
            if not self.config.add_skip_residue else skip_hidden_state
        last_hidden_state = last_hidden_state + skip_residue * grad_weight
        lm_logits = self.base.lm_head(last_hidden_state)
        nlls_skip = F.cross_entropy(
            input=lm_logits[:, :-1, :].contiguous().transpose(1, 2),
            target=labels[:, 1:].contiguous(),
            reduction='none'
        ).sum(dim=-1)

        if reduction == "mean" and self.config.add_skip_residue_contra:
            loss_skip_residue_contra = self.skip_residue_contra_loss(last_hidden_state, skip_hidden_state, labels)
            return nlls_skip.mean(), loss_skip_residue_contra

        if reduction == "mean":
            loss_lm_skip = nlls_skip.mean()
            return loss_lm_skip
        elif reduction == "none" or reduction is None:
            return nlls_skip
        else:
            raise NotImplementedError(f"reduction: {reduction}")

    def skip_residue_contra_loss(self, last_hidden_state, skip_hidden_state, labels):
        self.base.requires_grad_(False)
        in_batch_skip_nlls = []
        batch_size = last_hidden_state.shape[0]
        for bias in range(batch_size):
            biased_skip_hidden_state = torch.roll(skip_hidden_state, shifts=bias, dims=0)
            biased_skip_last_hidden_state = last_hidden_state.detach() + biased_skip_hidden_state
            biased_skip_lm_logits = self.base.lm_head(biased_skip_last_hidden_state)
            biased_skip_nlls = F.cross_entropy(
                input=biased_skip_lm_logits[:, :-1, :].contiguous().transpose(1, 2),
                target=labels[:, 1:].contiguous(),
                reduction='none'
            ).sum(dim=-1)
            in_batch_skip_nlls.append(biased_skip_nlls)
        in_batch_skip_nlls = torch.stack(in_batch_skip_nlls, dim=0)
        loss_skip_residue_contra = F.cross_entropy(
            input=-in_batch_skip_nlls.T,
            target=torch.zeros(batch_size, dtype=torch.int64, device=in_batch_skip_nlls.device),
            reduction='mean'
        )
        self.base.requires_grad_(not self.config.frozen_pretrained)
        return loss_skip_residue_contra

    def nlls(self, input_ps, input_ids, labels, skip_hidden_state=None):
        last_hidden_state = self.base.transformer(
            input_ids=input_ids,
            past_key_values=input_ps,
            return_dict=True,
        ).last_hidden_state
        if skip_hidden_state is not None:
            last_hidden_state = last_hidden_state + skip_hidden_state
        lm_logits = self.base.lm_head(last_hidden_state)
        nlls = F.cross_entropy(
            input=lm_logits[:, :-1, :].contiguous().transpose(1, 2),
            target=labels[:, 1:].contiguous(),
            reduction='none'
        ).sum(dim=-1)
        return nlls

    def lm_loss(self, input_ps, input_ids, labels):
        return self.nlls(input_ps, input_ids, labels).mean()

    def contra_loss(self, input_ps, input_ids, labels, positive_skip_zs=None):
        batch_size = labels.shape[0]
        prefix_length = torch.stack([torch.nonzero(labels[i, :] != -100)[0][0] for i in range(batch_size)], dim=0)
        assert torch.all(prefix_length == prefix_length[0]), f"unexpected data with variable prefix length: " \
                                                             f"{prefix_length}\n{labels.tolist()}"
        prefix_length = prefix_length[0].item()
        assert prefix_length > 0, f"unexpected data with no prefix:\n{labels.tolist()}"
        assert torch.all(input_ids[:, :prefix_length] == input_ids[0, :prefix_length]), \
            f"unexpected data with non-shared prefix:\n{input_ids[:, :prefix_length].tolist()}"

        prefix_length -= 1  # the labels will be shifted
        prefix_ids = input_ids[:, :prefix_length]
        input_ids = input_ids[:, prefix_length:]
        labels = labels[:, prefix_length:]

        prompted_prefix_outputs = self.base.transformer(
            input_ids=prefix_ids,
            past_key_values=input_ps,
            use_cache=True,
            return_dict=True
        )
        past_key_values = prompted_prefix_outputs.past_key_values

        in_batch_nlls = []
        for bias in range(batch_size):
            biased_past_key_values = [
                (torch.roll(past_keys, shifts=bias, dims=0),
                 torch.roll(past_values, shifts=bias, dims=0))
                for past_keys, past_values in past_key_values
            ]
            if positive_skip_zs is not None and bias == 0:
                skip_hidden_state = F.layer_norm(
                    self.skip_connection(positive_skip_zs.transpose(1, 2)),
                    normalized_shape=[self.config.n_embd],
                    eps=self.config.layer_norm_epsilon
                )[:, prefix_length:, :]
                skip_hidden_state = skip_hidden_state - skip_hidden_state.detach()
            else:
                skip_hidden_state = None
            in_batch_nlls.append(self.nlls(biased_past_key_values, input_ids, labels,
                                           skip_hidden_state=skip_hidden_state))
        in_batch_nlls = torch.stack(in_batch_nlls, dim=0)
        loss_contra = F.cross_entropy(
            input=-in_batch_nlls.T,
            target=torch.zeros(batch_size, dtype=torch.int64, device=in_batch_nlls.device),
            reduction='mean'
        )
        return loss_contra

    def forward(
        self,
        post_ids: torch.LongTensor,  # [batch, ans_seq_len]
        input_ids: torch.LongTensor,  # [batch, seq_len]
        labels: torch.LongTensor,  # [batch, seq_len]
        return_loss=True,
        **kwargs,
    ):
        if self.config.post_with_x:
            post_mean, post_logvar = self.latent_encoder(input_ids)
        else:
            post_mean, post_logvar = self.latent_encoder(post_ids)
        if self.config.with_bn:
            post_mean = self.bn(post_mean)

        prior_mean, prior_logvar = torch.zeros_like(post_mean), torch.zeros_like(post_logvar)
        loss_kld = dgkld(self.config, prior_mean, prior_logvar, post_mean, post_logvar)

        # only compute lm loss for the (currently) concerned output
        input_ps = self.latent_decoder(post_mean, post_logvar)
        # n_layer * [mini_many, num_p, n_embd] <- [mini_many, dim_z]

        if self.config.add_contra_loss and self.config.add_skip_connection:
            positive_skip_zs = sampling(post_mean, post_logvar, n_samples=labels.shape[1])
            loss_contra = self.contra_loss(input_ps, input_ids, labels, positive_skip_zs=positive_skip_zs)
            loss_lm = self.lm_loss(input_ps, input_ids, labels)
        elif self.config.add_contra_loss:
            loss_contra = self.contra_loss(input_ps, input_ids, labels, positive_skip_zs=None)
            loss_lm = self.lm_loss(input_ps, input_ids, labels)
        elif self.config.add_skip_connection:
            loss_contra = torch.zeros_like(input_ps[0][0].sum())
            loss_lm = self.skip_grad_lm_loss(post_mean, post_logvar, input_ps, input_ids, labels)
        else:
            loss_contra = torch.zeros_like(input_ps[0][0].sum())
            loss_lm = self.lm_loss(input_ps, input_ids, labels)

        if self.config.without_dg_kld:
            standard_kld = 0.5 * (post_mean.pow(2) + post_logvar.exp() - post_logvar - 1).sum(dim=1).mean(dim=0)
            loss_vae = loss_lm + standard_kld + loss_contra
        else:
            loss_vae = loss_lm + loss_kld + loss_contra

        return loss_vae, loss_lm, loss_kld, loss_contra

    def load_lpo_policy_latent_encoder(self, lpo_model_path):
        lpo_model = LPOModel(self.config)
        lpo_model.load_state_dict(load_checkpoint(lpo_model_path, device=self.device))
        self.latent_encoder.load_state_dict(lpo_model.policy_latent_encoder.state_dict())
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
        post_labels: torch.LongTensor = None,
        post_latent: Tuple[torch.FloatTensor] = None,
        standard_prior: bool = False,
        latent_sampling: bool = None,
        **kwargs,
    ):
        input_ids = kwargs["input_ids"]
        if latent_sampling is None:
            latent_sampling = self.config.use_standard_prior or standard_prior

        prior_mean, prior_logvar = self.latent_encoder(input_ids)
        if post_latent is not None:
            mean, logvar = post_latent
            mean = mean.to(device=prior_mean.device, dtype=prior_mean.dtype)
            logvar = logvar.to(device=prior_logvar.device, dtype=prior_logvar.dtype)
        elif post_labels is not None:
            mean, logvar = self.latent_encoder(post_labels)
        elif self.config.use_standard_prior or standard_prior:
            mean, logvar = torch.zeros_like(prior_mean), torch.zeros_like(prior_logvar)
        else:
            mean, logvar = prior_mean, prior_logvar

        if not latent_sampling:
            logvar.fill_(-100)

        input_ps = self.latent_decoder(mean, logvar)
        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids[:, :-1] != self.config.pad_token_id).long()
        ], dim=1)

        outputs = self.base.transformer(
            input_ids=input_ids[:, :-1],
            past_key_values=input_ps,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True
        )
        kwargs["past_key_values"] = outputs.past_key_values

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids != self.config.pad_token_id).long()
        ], dim=1)
        kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = True

        if self.config.add_skip_connection and self.config.add_skip_residue:
            if self.config.lm_sampling_times == 0:
                zs = mean.unsqueeze(-1)
            else:
                zs = sampling(mean, logvar, n_samples=1)
            skip_hidden_state = F.layer_norm(self.skip_connection(zs.transpose(1, 2)),
                                             normalized_shape=[self.config.n_embd],
                                             eps=self.config.layer_norm_epsilon)
            kwargs["skip_residue"] = skip_hidden_state

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
    def from_vae(cls, base_model_path, **kwargs):
        gpt2_vae = GPT2ForVAE.from_pretrained(base_model_path)
        config = gpt2_vae.config
        config.update(kwargs)
        config.use_standard_prior = True
        model = cls(config)
        model.policy_latent_encoder.load_state_dict(gpt2_vae.latent_encoder.state_dict())
        model.policy_latent_encoder.standardize()
        model.post_latent_encoder.load_state_dict(gpt2_vae.latent_encoder.state_dict())
        return model

    def forward(
        self,
        prompt_ids: torch.LongTensor,  # x
        chosen_ids: torch.LongTensor,  # y_w
        rejected_ids: torch.LongTensor,  # y_l
    ):
        prior_ids, chosen_post_ids, rejected_post_ids = prompt_ids, chosen_ids, rejected_ids
        # the latent distribution of pi_theta(z|x), pi_ref(z|x)
        # and approximation of pi_ref(z|x,y_w), pi_ref(z|x,y_l) (by their posterior distributions)
        policy_mean, policy_logvar = self.policy_latent_encoder(prior_ids)
        with torch.no_grad():
            if self.config.use_standard_prior:
                prior_mean, prior_logvar = torch.zeros_like(policy_mean), torch.zeros_like(policy_logvar)
            else:
                prior_mean, prior_logvar = self.prior_latent_encoder(prior_ids)
            if self.config.post_with_x:
                chosen_post_mean, chosen_post_logvar = self.post_latent_encoder(
                    torch.cat([prior_ids, chosen_post_ids], dim=1)
                )
                rejected_post_mean, rejected_post_logvar = self.post_latent_encoder(
                    torch.cat([prior_ids, rejected_post_ids], dim=1)
                )
            else:
                chosen_post_mean, chosen_post_logvar = self.post_latent_encoder(chosen_post_ids)
                rejected_post_mean, rejected_post_logvar = self.post_latent_encoder(rejected_post_ids)
        # batch_size, dim_z

        n_samples = self.config.lpo_sampling_times
        if self.config.expectation_lpo:
            zs = sampling(policy_mean, policy_logvar, n_samples=n_samples)
        else:
            prior_zs = sampling(prior_mean, prior_logvar, n_samples=n_samples)
            if self.config.lpo_post_sampling:
                chosen_zs = sampling(chosen_post_mean, chosen_post_logvar, n_samples=n_samples)
                rejected_zs = sampling(rejected_post_mean, rejected_post_logvar, n_samples=n_samples)
                zs = torch.cat([prior_zs, chosen_zs, rejected_zs], dim=-1)
            else:
                zs = prior_zs
        # batch_size, dim_z, n_samples

        # -log pi_ref(z_d|x,y_w)
        zs_marginal_nlls_given_chosen = -log_pdf(chosen_post_mean, chosen_post_logvar, zs)
        # -log pi_ref(z_d|x,y_l)
        zs_marginal_nlls_given_rejected = -log_pdf(rejected_post_mean, rejected_post_logvar, zs)
        # batch_size, dim_z, n_samples

        if self.config.marginal_lpo:
            batch_size, dim_z, n_samples = zs_marginal_nlls_given_chosen.shape
            zs_nlls_given_chosen = zs_marginal_nlls_given_chosen.view(batch_size * dim_z, n_samples)
            zs_nlls_given_rejected = zs_marginal_nlls_given_rejected.view(batch_size * dim_z, n_samples)
        else:
            # -log pi_ref(z|x,y_w)
            zs_nlls_given_chosen = zs_marginal_nlls_given_chosen.sum(dim=1)
            # -log pi_ref(z|x,y_l)
            zs_nlls_given_rejected = zs_marginal_nlls_given_rejected.sum(dim=1)

        # r(x,z) = (w_win*r_win + w_loss*r_loss) / (w_win + w_loss)
        # w \propto exp(-zs_nlls_given_chosen/rejected)
        importance_weights = F.softmax(
            torch.stack([-zs_nlls_given_chosen, -zs_nlls_given_rejected], dim=-1),
            dim=-1
        )
        win_lose_rewards = torch.Tensor([1, 0]).to(
            device=importance_weights.device, dtype=importance_weights.dtype
        )[None, None, :]
        latent_rewards = (importance_weights * win_lose_rewards).sum(dim=-1)
        score_zs = latent_rewards
        # batch_size, n_samples

        zs_marginal_refer_nlls = -log_pdf(prior_mean, prior_logvar, zs)
        zs_marginal_policy_nlls = -log_pdf(policy_mean, policy_logvar, zs)
        if self.config.marginal_lpo:
            zs_refer_nlls = zs_marginal_refer_nlls.view(batch_size * dim_z, n_samples)
            zs_policy_nlls = zs_marginal_policy_nlls.view(batch_size * dim_z, n_samples)
        else:
            zs_refer_nlls = zs_marginal_refer_nlls.sum(dim=1)
            zs_policy_nlls = zs_marginal_policy_nlls.sum(dim=1)
        # batch_size, n_samples

        if self.config.expectation_lpo:
            # maximize E_{z \sim \pi_{\theta}(z|x)} r(x,z) - \beta [log \pi_{\theta}(z|x) - log \pi_{ref}(z|x)]
            loss = self.config.beta * (zs_refer_nlls - zs_policy_nlls) - latent_rewards
            loss = loss.mean()
            acc = torch.zeros_like(loss)
            return loss, acc

        comparison_matrix = score_zs[:, :, None] - score_zs[:, None, :]
        # import pdb;pdb.set_trace()
        comparison_matrix = comparison_matrix.detach()
        # batch_size, n_samples, n_samples

        zs_policy_reward = (zs_refer_nlls.detach() - zs_policy_nlls)
        # batch_size, n_samples

        batch_size, n_samples = zs_policy_reward.shape

        zs_policy_reward_left = zs_policy_reward[:, :, None].repeat(1, 1, n_samples)
        zs_policy_reward_right = zs_policy_reward[:, None, :].repeat(1, n_samples, 1)
        # batch_size, n_samples, n_samples

        evidence_thresh = torch.quantile(comparison_matrix.view(batch_size, -1).float(),
                                         1 - self.config.lpo_evidence_ratio / 2, dim=1)
        left_win = comparison_matrix >= evidence_thresh[:, None, None]
        # batch_size, n_samples, n_samples
        comparison_mask = 1 - torch.eye(n_samples).float()
        comparison_mask = comparison_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        comparison_mask = comparison_mask.to(comparison_matrix.device)
        # batch_size, n_samples, n_samples
        comparison_mask = comparison_mask * left_win

        disadvantage = (zs_policy_reward_right - zs_policy_reward_left) * self.config.beta  # to minimize
        loss = -F.logsigmoid(-disadvantage)
        acc = (loss < 0.6931).float()
        loss = (loss * comparison_mask).sum() / comparison_mask.sum()
        acc = (acc * comparison_mask).sum() / comparison_mask.sum()

        return loss, acc


class DPOModel(GPT2PreTrainedModel):
    config_class = GPT2Config
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config):
        super().__init__(config)
        self.base = GPT2LMHeadModel(config)
        self.beta = config.beta

    def load_pretrained(self, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = GPT2LMHeadModel.from_pretrained(self.config.base_model_path)
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
        log_iw = refer_nlls.detach() - theta_nlls
        chosen_log_iw = log_iw[::2]
        rejected_log_iw = log_iw[1::2]
        disadvantage = (rejected_log_iw - chosen_log_iw) * self.beta
        loss = -F.logsigmoid(-disadvantage)
        acc = (loss < 0.6931).float()
        return loss.mean(), acc.mean()

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


class PTModel(GPT2PreTrainedModel):
    config_class = GPT2CVAEConfig
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: GPT2CVAEConfig):
        super().__init__(config)
        self.base = GPT2LMHeadModel(config)
        self.base.requires_grad_(False)
        self.hs = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(1, config.num_q, config.n_embd))
            for i in range(config.n_layer)
        ])
        for param in self.hs:
            param.data.normal_(mean=0.0, std=config.initializer_range)
        self.hs_dropout = nn.Dropout(p=0.1)
        self.lns = nn.ModuleList([nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
                                  for i in range(config.n_layer)])
        self.attns = nn.ModuleList([GPT2Attention(config, layer_idx=i)
                                    for i in range(config.n_layer)])
        self.attns.requires_grad_(False)
        self.beta = config.beta
        self.mode = "sft"

    def load_pretrained(self, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = GPT2LMHeadModel.from_pretrained(self.config.base_model_path)
        self.base.load_state_dict(pretrained_model.state_dict())
        for i in range(self.config.n_layer):
            self.attns[i].load_state_dict(pretrained_model.transformer.h[i].attn.state_dict())

    def switch_into_dpo_mode(self):
        self.mode = "dpo"
        self.refer_model = copy.deepcopy(self)
        self.refer_model.requires_grad_(False)
        self.refer_model.eval()

    def get_input_ps(self, input_ids):
        batch_size = input_ids.shape[0]
        past_key_values = []
        for i in range(self.config.n_layer):
            hidden_states, ln, attn = self.hs[i], self.lns[i], self.attns[i]
            hidden_states = hidden_states.repeat(batch_size, 1, 1)
            hidden_states = ln(self.hs_dropout(hidden_states))
            present = attn(
                hidden_states=hidden_states,
                use_cache=True,
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
            input_ps = self.get_input_ps(input_ids)
            return self.base(
                input_ids=input_ids,
                past_key_values=input_ps,
                labels=labels,
                return_dict=False
            )
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

        outputs = self.base.transformer(
            input_ids=input_ids[:, :-1],
            past_key_values=input_ps,
            use_cache=True,
            return_dict=True
        )
        kwargs["past_key_values"] = outputs.past_key_values

        attention_mask = torch.cat([
            torch.ones(input_ids.shape[0], self.config.num_p).long().to(input_ids.device),
            (input_ids != self.config.pad_token_id).long()
        ], dim=1)
        kwargs["attention_mask"] = attention_mask

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