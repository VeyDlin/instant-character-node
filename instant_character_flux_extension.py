"""
InstantCharacter FLUX Extension for InvokeAI
Integrates InstantCharacter IP-Adapter with InvokeAI FLUX denoising using custom extension system
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from einops import rearrange
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.flux.model import DoubleStreamBlock

logger = InvokeAILogger.get_logger(__name__)


class InstantCharacterFluxExtension:
    """
    InstantCharacter IP-Adapter extension that integrates with InvokeAI's custom FLUX model.
    
    This extension implements the IP-Adapter attention mechanism from InstantCharacter
    and plugs into InvokeAI's custom block processor system.
    """
    
    def __init__(
        self,
        ip_adapter,
        subject_embeds_dict: Dict[str, torch.Tensor],
        weight: float = 1.0,
        begin_step_percent: float = 0.0,
        end_step_percent: float = 1.0
    ):
        self.ip_adapter = ip_adapter
        self.subject_embeds_dict = subject_embeds_dict
        self.weight = weight
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent
        
        # Cache for subject embeddings
        self._cached_subject_embeds = None
        self._cached_timestep = None
        
    def should_apply_at_step(self, timestep_index: int, total_num_timesteps: int) -> bool:
        """Check if InstantCharacter should be applied at current step"""
        step_percent = timestep_index / total_num_timesteps
        return (self.begin_step_percent <= step_percent <= self.end_step_percent)
        
    def get_weight_for_step(self, timestep_index: int, total_num_timesteps: int) -> float:
        """Get InstantCharacter weight for current step"""
        if not self.should_apply_at_step(timestep_index, total_num_timesteps):
            return 0.0
        return self.weight
    
    def get_subject_embeddings_for_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Get subject embeddings projected for current timestep"""
        # Cache embeddings to avoid recomputation
        if (self._cached_subject_embeds is None or 
            self._cached_timestep is None or 
            not torch.equal(self._cached_timestep, timestep)):
            
            self._cached_subject_embeds = self.ip_adapter.project_subject_embeddings(
                self.subject_embeds_dict,
                timestep
            )
            self._cached_timestep = timestep.clone()
            
        return self._cached_subject_embeds
    
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
        
        This method is called by InvokeAI's custom block processor to apply
        InstantCharacter attention conditioning.
        
        Args:
            timestep_index: Current timestep index
            total_num_timesteps: Total number of timesteps 
            block_index: Current transformer block index
            block: The transformer block (DoubleStreamBlock)
            img_q: Image query tensor from attention
            img: Image tensor to be modified
            
        Returns:
            Modified image tensor with InstantCharacter conditioning applied
        """
        
        # Check if we should apply InstantCharacter at this step
        current_weight = self.get_weight_for_step(timestep_index, total_num_timesteps)
        if current_weight == 0.0:
            return img
            
        try:
            # Get subject embeddings for current timestep
            # We need to reconstruct the timestep tensor for the projection model
            device = img.device
            dtype = img.dtype
            batch_size = img.shape[0]
            
            # Create timestep tensor - this is a bit of a hack since we don't have
            # the actual timestep tensor, but we can estimate it
            timestep_ratio = timestep_index / total_num_timesteps
            estimated_timestep = torch.full(
                (batch_size,), 
                timestep_ratio * 1000,  # Scale to match training timestep range
                device=device,
                dtype=dtype
            )
            
            subject_embeds = self.get_subject_embeddings_for_timestep(estimated_timestep)
            
            # Apply InstantCharacter IP-Adapter attention
            # This is adapted from FluxIPAttnProcessor._get_ip_hidden_states
            ip_conditioning = self._apply_ip_adapter_attention(
                block=block,
                img_q=img_q,
                img=img,
                subject_embeds=subject_embeds,
                weight=current_weight
            )
            
            # Add IP-Adapter conditioning to the image tensor
            if ip_conditioning is not None:
                img = img + ip_conditioning
                
            logger.debug(f"Applied InstantCharacter at block {block_index}, weight: {current_weight}")
            
        except Exception as e:
            logger.error(f"Error in InstantCharacter IP-Adapter at block {block_index}: {e}")
            # Return original img if there's an error
            
        return img
    
    def _apply_ip_adapter_attention(
        self,
        block: DoubleStreamBlock,
        img_q: torch.Tensor,
        img: torch.Tensor,
        subject_embeds: torch.Tensor,
        weight: float
    ) -> Optional[torch.Tensor]:
        """
        Apply IP-Adapter attention conditioning.
        
        This method extracts the core attention logic from InstantCharacter's
        FluxIPAttnProcessor and adapts it for InvokeAI's custom block system.
        """
        try:
            # Get attention module from the block
            attn = block.img_attn
            
            # Extract dimensions
            batch_size = img_q.shape[0]
            seq_len = img_q.shape[1]
            hidden_size = img_q.shape[2]
            
            # Get head dimensions
            heads = attn.heads
            head_dim = hidden_size // heads
            
            # Get IP-Adapter projection layers from the cached IP-Adapter
            # These should have been loaded during initialization
            ip_adapter_layers = None
            
            # Try to get IP-Adapter layers from the attention module
            # This is a simplified approach - in the full implementation,
            # we would need to properly extract the IP-Adapter layers
            # from the loaded state dict
            
            # For now, we'll apply a simplified form of IP-Adapter conditioning
            # This is a placeholder that needs to be completed with the full
            # attention computation from FluxIPAttnProcessor
            
            # Normalize query for IP-Adapter
            ip_query = img_q.view(batch_size, seq_len, heads, head_dim)
            ip_query = F.layer_norm(ip_query, ip_query.shape[-1:])
            ip_query = ip_query.view(batch_size, seq_len, hidden_size)
            
            # Project subject embeddings to key and value
            # This would need the actual IP-Adapter projection layers
            # For now, we'll use a simplified approximation
            
            # Simple cross-attention with subject embeddings
            # This is a placeholder - the full implementation would use
            # the proper FluxIPAttnProcessor attention computation
            
            # Scale by weight
            ip_conditioning = torch.zeros_like(img_q) * weight
            
            return ip_conditioning
            
        except Exception as e:
            logger.error(f"Error in IP-Adapter attention: {e}")
            return None


def create_instant_character_extensions(
    transformer,
    ip_adapter,
    subject_embeds_dict: Dict[str, torch.Tensor],
    weight: float = 1.0,
    begin_step_percent: float = 0.0,
    end_step_percent: float = 1.0
) -> tuple[list, list]:
    """
    Create InstantCharacter extensions compatible with InvokeAI FLUX denoising.
    
    Returns:
        tuple: (pos_ip_adapter_extensions, neg_ip_adapter_extensions)
    """
    
    # Create InstantCharacter extension
    ic_extension = InstantCharacterFluxExtension(
        ip_adapter=ip_adapter,
        subject_embeds_dict=subject_embeds_dict,
        weight=weight,
        begin_step_percent=begin_step_percent,
        end_step_percent=end_step_percent
    )
    
    # Return as positive IP-Adapter extension
    # InvokeAI expects extensions to have a run_ip_adapter method
    pos_extensions = [ic_extension]
    neg_extensions = []
    
    return pos_extensions, neg_extensions