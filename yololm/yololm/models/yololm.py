import os
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from yololm.utils import add_spatial_token
from ultralytics import YOLOWorld
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

class YOLOLMConfig(LlamaConfig):
    model_type = "yololm"

class YOLOLM(LlamaModel):
    config_class = YOLOLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.in_channels = getattr(config, 'in_channels', 512)
        
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            self.vision_tower = YOLOWorld(config.mm_vision_tower).model

        self.mm_projector = torch.nn.Linear(self.in_channels, config.hidden_size)
    
    def initialize_vision_modules(self, vision_tower):
        self.config.mm_vision_tower = vision_tower
        
        if not hasattr(self, 'vision_tower'):
            vision_tower = YOLOWorld(vision_tower).model
        else:
            vision_tower = self.vision_tower
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(dtype=torch.float16)
        self.vision_tower = vision_tower

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            visuals=None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        
        if vision_tower is not None and (input_ids.shape[
                                             1] != 1 or self.training) and images is not None:
            assert(past_key_values is None)
            mlvl_spi_features = []
            with torch.no_grad():
                vision_tower.eval()
                for index, visual in enumerate(visuals):
                    mlvl_spi_features.append(vision_tower.get_visual_pe(images[index].unsqueeze(0), visual=visual.unsqueeze(0)))
                    
            mlvl_spi_features = torch.cat(mlvl_spi_features, dim=1).squeeze(0)
            mlvl_spi_features = self.mm_projector(mlvl_spi_features)
            
            bbox_token_id = self.tokenizer.convert_tokens_to_ids('<bbox>')

            spi_mask = (input_ids == bbox_token_id)
            inputs_embeds[spi_mask] = mlvl_spi_features.to(inputs_embeds.dtype)  # first 16 tokens are for vision

        return super().forward(
            input_ids=None, attention_mask=attention_mask,
            position_ids = position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class YOLOForCausalLM(LlamaForCausalLM):
    config_class = YOLOLMConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = YOLOLM(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            images=None,
            visuals=None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        self.model.orig_forward = self.model.forward
        self.model.forward = partial(self.model.orig_forward,
                                     visuals=visuals, images=images)
        
        outputs = super().forward(
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
        )
        
        self.model.forward = self.model.orig_forward
        return outputs

    def get_model(self):
        return self.model

    def initialize_vision_tokenizer(self, tokenizer):
        tokenizer = add_spatial_token(tokenizer)
        self.resize_token_embeddings(len(tokenizer))
        for m in self.modules():
            m.tokenizer = tokenizer

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
AutoConfig.register("yololm", YOLOLMConfig)
AutoModelForCausalLM.register(YOLOLMConfig, YOLOForCausalLM)