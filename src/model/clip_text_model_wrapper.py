import torch
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class CLIPTextModelWrapper(CLIPTextModel):
    """
    Arc2Face-compatible CLIP text encoder wrapper.
    Adds:
      - return_token_embs=True: return token embeddings before transformer
      - input_token_embs=<tensor>: inject precomputed token embeddings
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_token_embs=None,
        return_token_embs=False,
    ):
        if return_token_embs:
            if input_ids is None:
                raise ValueError("input_ids is required when return_token_embs=True")
            return self.text_model.embeddings.token_embedding(input_ids)

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.text_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.text_model.config.output_hidden_states
        )

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.text_model.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_token_embs,
        )

        bsz, seq_len = input_shape
        # Build CLIP causal mask: block attention to future tokens.
        min_val = torch.finfo(hidden_states.dtype).min
        causal_attention_mask = torch.full(
            (bsz, 1, seq_len, seq_len),
            fill_value=min_val,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        causal_attention_mask = torch.triu(causal_attention_mask, diagonal=1)
        if attention_mask is not None:
            # Convert [B,S] with 1=keep,0=mask to additive [B,1,1,S] mask.
            attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * min_val

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        if self.text_model.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                (
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                    == self.text_model.eos_token_id
                )
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
