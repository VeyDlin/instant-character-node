"""
InstantCharacter extension for InvokeAI following XLabs IP-Adapter pattern
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from einops import rearrange
import torch.nn.functional as F

from invokeai.backend.flux.modules.layers import DoubleStreamBlock, SingleStreamBlock
from invokeai.backend.util.logging import InvokeAILogger

from .InstantCharacter.models.attn_processor import FluxIPAttnProcessor
from .InstantCharacter.models.resampler import CrossLayerCrossScaleProjector
from .InstantCharacter.models.norm_layer import RMSNorm

logger = InvokeAILogger.get_logger(__name__)


class InstantCharacterModel(nn.Module):
    """InstantCharacter model compatible with InvokeAI's IP-Adapter system"""
    
    def __init__(
        self,
        num_double_blocks: int = 19,
        num_single_blocks: int = 38,
        hidden_size: int = 3072,
        ip_hidden_states_dim: int = 4096,
        nb_token: int = 1024,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Create IP-Adapter processors for double stream blocks only
        self.ip_double_blocks = nn.ModuleList([
            FluxIPAttnProcessor(
                hidden_size=hidden_size,
                ip_hidden_states_dim=ip_hidden_states_dim,
            ) for _ in range(num_double_blocks)
        ])
        
        # CrossLayerCrossScaleProjector for time-conditioned projection
        self.image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536,
            num_attention_heads=42,
            attention_head_dim=64,
            cross_attention_dim=1152 + 1536,
            num_layers=4,
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=nb_token,
            embedding_dim=1152 + 1536,
            output_dim=4096,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        
    def load_state_dict_from_file(self, state_dict_path: str):
        """Load InstantCharacter weights"""
        state_dict = torch.load(state_dict_path, map_location="cpu")
        
        # Load IP-Adapter weights for double blocks only
        ip_adapter_state = state_dict["ip_adapter"]
        for i, block in enumerate(self.ip_double_blocks):
            block_state = {
                k.replace(f"{i}.", ""): v 
                for k, v in ip_adapter_state.items() 
                if k.startswith(f"{i}.")
            }
            if block_state:
                block.load_state_dict(block_state, strict=True)
        
        # Load projection model weights
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        logger.info("Loaded InstantCharacter model weights")


class InstantCharacterExtension:
    """Extension that integrates InstantCharacter with InvokeAI's denoising"""
    
    def __init__(
        self,
        model: InstantCharacterModel,
        subject_embeds_dict: Dict[str, torch.Tensor],
        timesteps: list[float],
        weight: float = 1.0,
        begin_step_percent: float = 0.0,
        end_step_percent: float = 1.0,
    ):
        self._model = model
        self._subject_embeds_dict = subject_embeds_dict
        self._timesteps = timesteps
        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent
        
        # Cache for projected embeddings
        self._projected_embeds_cache: Dict[int, torch.Tensor] = {}
        
    def _get_weight(self, timestep_index: int, total_num_timesteps: int) -> float:
        """Get weight for current timestep"""
        step_percent = timestep_index / total_num_timesteps
        if self._begin_step_percent <= step_percent <= self._end_step_percent:
            return self._weight
        return 0.0
        
    def _get_projected_embeds(self, timestep_index: int) -> torch.Tensor:
        """Get time-conditioned subject embeddings (with caching)"""
        if timestep_index in self._projected_embeds_cache:
            return self._projected_embeds_cache[timestep_index]
            
        timestep = self._timesteps[timestep_index]
        timestep_tensor = torch.tensor([timestep / 1000.0], device=self._model.device, dtype=self._model.dtype)
        
        with torch.no_grad():
            projected = self._model.image_proj_model(
                low_res_shallow=self._subject_embeds_dict['image_embeds_low_res_shallow'],
                low_res_deep=self._subject_embeds_dict['image_embeds_low_res_deep'],
                high_res_deep=self._subject_embeds_dict['image_embeds_high_res_deep'],
                timesteps=timestep_tensor,
                need_temb=True
            )[0]
            
        self._projected_embeds_cache[timestep_index] = projected
        return projected
        
    def run_ip_adapter(
        self,
        timestep_index: int,
        total_num_timesteps: int,
        block_index: int,
        block: DoubleStreamBlock,
        img_q: torch.Tensor,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """Apply InstantCharacter IP-Adapter to double stream block"""
        
        weight = self._get_weight(timestep_index, total_num_timesteps)
        if weight < 1e-6:
            return img
            
        # Get IP-Adapter processor for this block
        ip_processor = self._model.ip_double_blocks[block_index]
        
        # Get projected subject embeddings
        subject_embeds = self._get_projected_embeds(timestep_index)
        
        # Apply InstantCharacter attention logic
        # Based on FluxIPAttnProcessor._get_ip_hidden_states
        
        # Normalize query
        ip_query = rearrange(img_q, 'B H L D -> B L (H D)')
        ip_query = rearrange(ip_query, 'B L (H D) -> B H L D', H=block.num_heads)
        ip_query = ip_processor.norm_ip_q(ip_query)
        ip_query = rearrange(ip_query, 'B H L D -> (B H) L D')
        
        # Project subject embeddings to K and V
        ip_key = ip_processor.to_k_ip(subject_embeds)
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=block.num_heads)
        ip_key = ip_processor.norm_ip_k(ip_key)
        ip_key = rearrange(ip_key, 'B H L D -> (B H) L D')
        
        ip_value = ip_processor.to_v_ip(subject_embeds)
        ip_value = rearrange(ip_value, 'B L (H D) -> (B H) L D', H=block.num_heads)
        
        # Compute attention
        ip_query_attn = rearrange(ip_query, '(B H) L D -> B H L D', H=block.num_heads)
        ip_key_attn = rearrange(ip_key, '(B H) L D -> B H L D', H=block.num_heads)
        ip_value_attn = rearrange(ip_value, '(B H) L D -> B H L D', H=block.num_heads)
        
        ip_hidden_states = F.scaled_dot_product_attention(
            ip_query_attn.to(ip_value_attn.dtype),
            ip_key_attn.to(ip_value_attn.dtype),
            ip_value_attn,
            dropout_p=0.0,
            is_causal=False
        )
        
        ip_hidden_states = rearrange(ip_hidden_states, 'B H L D -> B L (H D)')
        ip_hidden_states = ip_hidden_states.to(img.dtype)
        
        # Apply weighted addition
        img = img + weight * ip_hidden_states
        
        return img