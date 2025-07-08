"""
InstantCharacter FLUX Extension for InvokeAI
Properly integrates InstantCharacter IP-Adapter with InvokeAI's custom FLUX architecture
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from einops import rearrange
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.flux.model import DoubleStreamBlock

logger = InvokeAILogger.get_logger(__name__)


class InstantCharacterFluxExtension:
    """
    InstantCharacter IP-Adapter extension that properly integrates with InvokeAI's custom FLUX model.
    
    This extension implements InstantCharacter's IP-Adapter attention mechanism
    and plugs into InvokeAI's custom block processor system at the correct integration points.
    """
    
    def __init__(
        self,
        ip_adapter_layers: Dict[str, Any],
        image_proj_model,
        subject_embeds_dict: Dict[str, torch.Tensor],
        timesteps: List[float],
        weight: float = 1.0,
        begin_step_percent: float = 0.0,
        end_step_percent: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        self.ip_adapter_layers = ip_adapter_layers
        self.image_proj_model = image_proj_model
        self.weight = weight
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent
        self.device = device
        self.dtype = dtype
        
        # Precompute subject embeddings for all timesteps
        self._precompute_subject_embeddings(subject_embeds_dict, timesteps)
        
    def _precompute_subject_embeddings(self, subject_embeds_dict: Dict[str, torch.Tensor], timesteps: List[float]):
        """Precompute time-conditioned subject embeddings for all timesteps"""
        self._subject_embeddings = []
        
        logger.debug(f"Precomputing subject embeddings for {len(timesteps)} timesteps")
        
        for timestep in timesteps:
            # Convert timestep to tensor with proper scaling (like in original pipeline)
            timestep_tensor = torch.tensor([timestep / 1000.0], device=self.device, dtype=self.dtype)
            
            # Project subject embeddings with time conditioning using CrossLayerCrossScaleProjector
            with torch.inference_mode():
                subject_embeds = self.image_proj_model(
                    low_res_shallow=subject_embeds_dict['image_embeds_low_res_shallow'],
                    low_res_deep=subject_embeds_dict['image_embeds_low_res_deep'],
                    high_res_deep=subject_embeds_dict['image_embeds_high_res_deep'],
                    timesteps=timestep_tensor,
                    need_temb=True
                )[0]  # Extract first element from tuple
                
            self._subject_embeddings.append(subject_embeds)
            
        
    def should_apply_at_step(self, timestep_index: int, total_num_timesteps: int) -> bool:
        """Check if InstantCharacter should be applied at current step"""
        step_percent = timestep_index / total_num_timesteps
        return (self.begin_step_percent <= step_percent <= self.end_step_percent)
        
    def get_weight_for_step(self, timestep_index: int, total_num_timesteps: int) -> float:
        """Get InstantCharacter weight for current step"""
        if not self.should_apply_at_step(timestep_index, total_num_timesteps):
            return 0.0
        return self.weight
    
    def run_ip_adapter(
        self,
        timestep_index: int,
        total_num_timesteps: int,
        block_index: int,
        block: DoubleStreamBlock,
        img_q: torch.Tensor,
        img: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run InstantCharacter IP-Adapter processing for current block.
        
        This method is called by InvokeAI's CustomDoubleStreamBlockProcessor
        for each double stream block during the denoising process.
        
        Args:
            timestep_index: Current timestep index (0 to total_num_timesteps-1)
            total_num_timesteps: Total number of timesteps
            block_index: Current transformer block index (0 to 18 for FLUX.1-dev)
            block: The DoubleStreamBlock instance
            img_q: Image query tensor from attention computation [B, H, L, D]
            img: Image tensor to be modified [B, L, D]
            
        Returns:
            Modified image tensor with InstantCharacter conditioning applied
        """
        
        # Check if we should apply InstantCharacter at this step
        current_weight = self.get_weight_for_step(timestep_index, total_num_timesteps)
        if current_weight == 0.0:
            return img
            
        # Validate inputs
        if timestep_index >= len(self._subject_embeddings):
            logger.error(f"No subject embeddings for timestep {timestep_index}/{len(self._subject_embeddings)}")
            return img
            
        try:
            # Get precomputed subject embeddings for current timestep
            subject_embeds = self._subject_embeddings[timestep_index]
            
            # Get IP-Adapter layer for this block
            layer_key = f"layer_{block_index}"
            if layer_key not in self.ip_adapter_layers:
                # Try alternative mapping - check available layers
                available_layers = list(self.ip_adapter_layers.keys())
                if block_index < len(available_layers):
                    layer_key = available_layers[block_index]
                else:
                    logger.debug(f"No IP-Adapter layer for block {block_index}, skipping")
                    return img
                    
            ip_layer = self.ip_adapter_layers[layer_key]
            
            # Apply InstantCharacter IP-Adapter attention
            ip_conditioning = self._apply_instant_character_attention(
                ip_layer=ip_layer,
                block=block,
                img_q=img_q,
                img=img,
                subject_embeds=subject_embeds,
                weight=current_weight
            )
            
            # Add IP-Adapter conditioning to the image tensor
            if ip_conditioning is not None:
                img = img + ip_conditioning
                logger.debug(f"Applied InstantCharacter at block {block_index}, timestep {timestep_index}, weight: {current_weight:.3f}")
                
        except Exception as e:
            logger.error(f"Error in InstantCharacter IP-Adapter at block {block_index}: {e}")
            import traceback
            traceback.print_exc()
            
        return img
    
    def _apply_instant_character_attention(
        self,
        ip_layer,
        block: DoubleStreamBlock,
        img_q: torch.Tensor,
        img: torch.Tensor,
        subject_embeds: torch.Tensor,
        weight: float
    ) -> Optional[torch.Tensor]:
        """
        Apply InstantCharacter IP-Adapter attention conditioning.
        
        This implements the core logic from FluxIPAttnProcessor._get_ip_hidden_states
        adapted for InvokeAI's architecture.
        """
        try:
            # Get attention module from the block
            attn = block.img_attn
            
            # Extract tensor dimensions
            batch_size, seq_len, hidden_size = img.shape
            heads = attn.heads
            head_dim = hidden_size // heads
            
            # Validate subject embeddings shape
            if subject_embeds.dim() != 3:  # Should be [B, num_tokens, embed_dim]
                logger.error(f"Invalid subject embeddings shape: {subject_embeds.shape}")
                return None
                
            # Reshape img_q from [B, H, L, D] to [B, L, H*D] to match img tensor format
            if img_q.dim() == 4:
                img_q_reshaped = rearrange(img_q, 'b h l d -> b l (h d)')
            else:
                img_q_reshaped = img_q
                
            # === InstantCharacter IP-Adapter Attention Logic ===
            # This follows the exact pattern from FluxIPAttnProcessor._get_ip_hidden_states
            
            # 1. Normalize query for IP-Adapter
            # Reshape to heads format for normalization
            ip_query = rearrange(img_q_reshaped, 'b l (h d) -> b h l d', h=heads)
            ip_query = ip_layer.norm_ip_q(ip_query)  # RMSNorm normalization
            ip_query = rearrange(ip_query, 'b h l d -> (b h) l d')  # Flatten for attention
            
            # 2. Project subject embeddings to key and value
            ip_key = ip_layer.to_k_ip(subject_embeds)  # [B, num_tokens, hidden_size]
            ip_key = rearrange(ip_key, 'b l (h d) -> b h l d', h=heads)
            ip_key = ip_layer.norm_ip_k(ip_key)  # RMSNorm normalization
            ip_key = rearrange(ip_key, 'b h l d -> (b h) l d')  # Flatten for attention
            
            ip_value = ip_layer.to_v_ip(subject_embeds)  # [B, num_tokens, hidden_size]
            ip_value = rearrange(ip_value, 'b l (h d) -> (b h) l d', h=heads)
            
            # 3. Compute scaled dot product attention between query and IP projections
            # This is the core cross-attention computation from InstantCharacter
            ip_query_for_attn = rearrange(ip_query, '(b h) l d -> b h l d', h=heads)
            ip_key_for_attn = rearrange(ip_key, '(b h) l d -> b h l d', h=heads)
            ip_value_for_attn = rearrange(ip_value, '(b h) l d -> b h l d', h=heads)
            
            # Apply scaled dot product attention
            ip_hidden_states = F.scaled_dot_product_attention(
                ip_query_for_attn.to(ip_value_for_attn.dtype),
                ip_key_for_attn.to(ip_value_for_attn.dtype),
                ip_value_for_attn,
                dropout_p=0.0,
                is_causal=False
            )
            
            # 4. Reshape back to match img tensor format and apply weight
            ip_hidden_states = rearrange(ip_hidden_states, 'b h l d -> b l (h d)')
            ip_hidden_states = ip_hidden_states.to(img.dtype)
            
            # Scale by weight (this is the subject_scale from the original)
            ip_conditioning = ip_hidden_states * weight
            
            return ip_conditioning
            
        except Exception as e:
            logger.error(f"Error in InstantCharacter attention computation: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_instant_character_extensions(
    ip_adapter_layers: Dict[str, Any],
    image_proj_model,
    subject_embeds_dict: Dict[str, torch.Tensor],
    timesteps: List[float],
    weight: float = 1.0,
    begin_step_percent: float = 0.0,
    end_step_percent: float = 1.0,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> tuple[list, list]:
    """
    Create InstantCharacter extensions compatible with InvokeAI FLUX denoising.
    
    Args:
        ip_adapter_layers: Dictionary of IP-Adapter layers (from InstantCharacterIPAdapter)
        image_proj_model: CrossLayerCrossScaleProjector for time-conditioned projection
        subject_embeds_dict: Dictionary of encoded subject image embeddings
        timesteps: List of timesteps for the denoising process
        weight: IP-Adapter weight (subject_scale from original)
        begin_step_percent: Start applying at this percentage of steps
        end_step_percent: Stop applying at this percentage of steps
        device: Device for tensor operations
        dtype: Data type for tensor operations
    
    Returns:
        tuple: (pos_ip_adapter_extensions, neg_ip_adapter_extensions)
    """
    
    # Create InstantCharacter extension
    ic_extension = InstantCharacterFluxExtension(
        ip_adapter_layers=ip_adapter_layers,
        image_proj_model=image_proj_model,
        subject_embeds_dict=subject_embeds_dict,
        timesteps=timesteps,
        weight=weight,
        begin_step_percent=begin_step_percent,
        end_step_percent=end_step_percent,
        device=device,
        dtype=dtype
    )
    
    # Return as positive IP-Adapter extension
    # InvokeAI's CustomDoubleStreamBlockProcessor expects extensions with run_ip_adapter method
    pos_extensions = [ic_extension]
    neg_extensions = []  # InstantCharacter doesn't use negative extensions
    
    return pos_extensions, neg_extensions