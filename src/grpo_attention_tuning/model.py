import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple

from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from noisy_layer import replace_with_noisy_linear

class VanillaRoPE(nn.Module):
    def __init__(self, hidden_size, device=None):
        super(VanillaRoPE, self).__init__()
        base = 1000
        dim = hidden_size
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, end_k = -1):
        super().__init__()
        self.hidden_size = hidden_size
        # need the end_k states to calculate
        # if set to -1, all of hidden_states needed
        self.end_k = end_k

        # 定义Q, K, V的线性变换层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale_score = 1.0/(self.hidden_size ** 0.5)
        self.rope = VanillaRoPE(hidden_size)

    @staticmethod
    def mask_to_weights(attention_mask, thought_id_idx, end_k=-1):
        if thought_id_idx is not None and attention_mask.size(0) != thought_id_idx.size(0):
            raise ValueError("attention_mask must have the same size as thought_id_idx")

        # useless
        if end_k != -1:
            # 把thought_id前end_k个位置的内容考虑进去
            # 这里的代码有一点问题：也就是我们认为end_k不会长过前面的句子长度
            # 否则会把padding的部分置为1
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                start_idx = max(0, idx - end_k)
                attention_mask[i].zero_()
                attention_mask[i, start_idx:idx] = 1
        else:
            # 把thought_id前所有的内容都考虑进去 (要屏蔽padding和thought_id后面的部分)
            # 这里想一下
            # 在训练的时候，padding在右侧
            # 在推理的时候，padding在左侧
            # 而在推理的时候，我们传进来的attention_mask实际上是正确的，所以不需要全部置0
            # 训练时候，代码可以改为: attention_mask[i, idx:] = 0
            # 推理时候, idx == seq_len, 也是可以和训练时候一样去改，所以修改
            for i in range(attention_mask.size(0)):
                idx = thought_id_idx[i].item()
                attention_mask[i, idx:] = 0

        # Convert mask to weights: 1 -> 0, 0 -> -inf
        attention_weight = torch.zeros_like(attention_mask, dtype=torch.float32)
        attention_weight = attention_weight.masked_fill(~attention_mask.bool(), float('-inf'))
        return attention_weight

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                thought_id_idx) -> torch.Tensor:
        # hidden_states = hidden_states.to(torch.float32)
        attention_mask = SelfAttentionLayer.mask_to_weights(attention_mask.clone(), thought_id_idx, self.end_k)

        cos, sin = self.rope(
            hidden_states,
            position_ids=torch.arange(0, hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        )

        batch_size, seq_len, _ = hidden_states.shape

        # 生成Q, K, V
        Q = self.query(hidden_states)  # [batch_size, seq_len, embed_dim]
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # 加入位置编码
        Q, K = apply_rotary_pos_emb(Q, K, cos.squeeze(0), sin.squeeze(0), unsqueeze_dim=0)

        # 确定目标位置索引
        if thought_id_idx is None:
            # 取每个样本的最后一个位置（seq_len - 1）
            indices = torch.full((batch_size,), seq_len - 1, device=hidden_states.device)
        else:
            indices = thought_id_idx - 1
            if (indices < 0).any() or (indices >= seq_len).any():
                raise ValueError("thought_id_idx-1 exceeds valid sequence indices")

        # 提取目标位置的Q向量 [batch_size, 1, embed_dim]
        Q_selected = Q[torch.arange(batch_size), indices].unsqueeze(1)

        # 计算注意力分数 [batch_size, 1, seq_len]
        attn_scores = torch.matmul(Q_selected, K.transpose(-2, -1)) * self.scale_score

        # 应用attention_mask（形状需匹配为[batch_size, 1, seq_len]）
        attn_scores += attention_mask.unsqueeze(1)  # 广播至[batch_size, 1, seq_len]

        # 计算注意力权重与输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        output_selected = torch.matmul(attn_weights, V)  # [batch_size, 1, embed_dim]
        output = output_selected.squeeze(1) # [batch_size, embed_dim]
        return output



class LatentModel(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.attention = SelfAttentionLayer(config.hidden_size)

    def generate_embs(self, input_ids, attention_mask):
        output_mid = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # generate embeddings
        thought_ids = self.model.embed_tokens.num_embeddings - 1  # thought id flag
        where_thought_ids = torch.nonzero(input_ids == thought_ids)
        hidden_states = output_mid['hidden_states']
        input_embs = self.model.embed_tokens(input_ids)

        input_embs[where_thought_ids[:, 0], where_thought_ids[:, 1]] = self.attention(
            hidden_states=hidden_states[-1], attention_mask=attention_mask,
            thought_id_idx=where_thought_ids[:, 1],
        ).to(input_embs.dtype)
        return input_embs


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if input_ids is not None and inputs_embeds is None and input_ids.size() == attention_mask.size():
            inputs_embeds = self.generate_embs(input_ids, attention_mask)
            input_ids = None
        # elif input_ids is not None and inputs_embeds is None and input_ids.size()!=attention_mask.size():
            # 这个时候是在做生成任务，已经有了cache所以用input_ids即可
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


# class NoisyLatentModel(LatentModel):
#     def __init__(self, config):
#         super().__init__(config)
#         replace_with_noisy_linear(self.attention)
