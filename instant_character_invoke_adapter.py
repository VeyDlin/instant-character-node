"""
Adapter to use InstantCharacter with InvokeAI's IP-Adapter loading system
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from invokeai.backend.util.logging import InvokeAILogger
from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.backend.model_manager.config import AnyModelConfig

from .instant_character_invoke_extension import InstantCharacterModel, InstantCharacterExtension
from .instant_character_models import InstantCharacterImageEncoder

logger = InvokeAILogger.get_logger(__name__)


class InstantCharacterInvokeAdapter:
    """Adapter that makes InstantCharacter compatible with InvokeAI's model loading"""
    
    @staticmethod
    def create_model_config() -> Dict[str, Any]:
        """Create config for InstantCharacter model"""
        return {
            "format": "checkpoint",
            "type": "ip_adapter",
            "base": "flux",
        }
    
    @staticmethod
    def load_model(
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16
    ) -> InstantCharacterModel:
        """Load InstantCharacter model"""
        model = InstantCharacterModel(
            num_double_blocks=19,  # FLUX has 19 double stream blocks
            num_single_blocks=38,  # FLUX has 38 single stream blocks
            hidden_size=3072,      # FLUX hidden size
            ip_hidden_states_dim=4096,  # T5 hidden size
            nb_token=1024,
            device=device,
            dtype=dtype
        )
        
        # Move to device and dtype
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        # Load weights
        model.load_state_dict_from_file(model_path)
        
        return model
    
    @staticmethod
    def encode_subject_image(
        context: InvocationContext,
        image_encoder: InstantCharacterImageEncoder,
        image: Image.Image,
        device: torch.device,
        dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """Encode subject image using SigLIP and DINOv2"""
        
        # Prepare multi-resolution images
        object_image_pil_low_res = [image.resize((384, 384))]
        object_image_pil_high_res = image.resize((768, 768))
        object_image_pil_high_res = [
            object_image_pil_high_res.crop((0, 0, 384, 384)),
            object_image_pil_high_res.crop((384, 0, 768, 384)),
            object_image_pil_high_res.crop((0, 384, 384, 768)),
            object_image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res)
        
        # Process low resolution
        siglip_low_res = image_encoder.siglip_processor(
            images=object_image_pil_low_res, 
            return_tensors="pt"
        ).pixel_values.to(device, dtype=dtype)
        siglip_embeds_low = image_encoder.encode_siglip_image(siglip_low_res)
        
        dinov2_low_res = image_encoder.dinov2_processor(
            images=object_image_pil_low_res,
            return_tensors="pt"
        ).pixel_values.to(device, dtype=dtype)
        dinov2_embeds_low = image_encoder.encode_dinov2_image(dinov2_low_res)
        
        # Combine low resolution
        image_embeds_low_res_deep = torch.cat([siglip_embeds_low[0], dinov2_embeds_low[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_embeds_low[1], dinov2_embeds_low[1]], dim=2)
        
        # Process high resolution
        from einops import rearrange
        
        siglip_high_res = image_encoder.siglip_processor(
            images=object_image_pil_high_res,
            return_tensors="pt"
        ).pixel_values.to(device, dtype=dtype)
        siglip_high_res = siglip_high_res[None]
        siglip_high_res = rearrange(siglip_high_res, 'b n c h w -> (b n) c h w')
        siglip_embeds_high = image_encoder.encode_siglip_image(siglip_high_res)
        siglip_high_res_deep = rearrange(siglip_embeds_high[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        
        dinov2_high_res = image_encoder.dinov2_processor(
            images=object_image_pil_high_res,
            return_tensors="pt"
        ).pixel_values.to(device, dtype=dtype)
        dinov2_high_res = dinov2_high_res[None]
        dinov2_high_res = rearrange(dinov2_high_res, 'b n c h w -> (b n) c h w')
        dinov2_embeds_high = image_encoder.encode_dinov2_image(dinov2_high_res)
        dinov2_high_res_deep = rearrange(dinov2_embeds_high[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        
        # Combine high resolution
        image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)
        
        return {
            "image_embeds_low_res_shallow": image_embeds_low_res_shallow,
            "image_embeds_low_res_deep": image_embeds_low_res_deep,
            "image_embeds_high_res_deep": image_embeds_high_res_deep,
        }
    
    @staticmethod
    def create_extension(
        model: InstantCharacterModel,
        subject_embeds_dict: Dict[str, torch.Tensor],
        timesteps: List[float],
        weight: float = 1.0,
        begin_step_percent: float = 0.0,
        end_step_percent: float = 1.0,
    ) -> Tuple[List[InstantCharacterExtension], List]:
        """Create InstantCharacter extensions for InvokeAI denoising"""
        
        extension = InstantCharacterExtension(
            model=model,
            subject_embeds_dict=subject_embeds_dict,
            timesteps=timesteps,
            weight=weight,
            begin_step_percent=begin_step_percent,
            end_step_percent=end_step_percent,
        )
        
        # Return as positive extension, no negative extension
        return [extension], []