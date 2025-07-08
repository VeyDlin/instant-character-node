"""
InstantCharacter models and components adapted for InvokeAI
"""
import torch
from torch import nn
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
from invokeai.backend.util.logging import InvokeAILogger

from .InstantCharacter.models.attn_processor import FluxIPAttnProcessor
from .InstantCharacter.models.resampler import CrossLayerCrossScaleProjector

logger = InvokeAILogger.get_logger(__name__)


class InstantCharacterImageEncoder:
    """Manages image encoders for InstantCharacter"""
    
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.siglip_encoder = None
        self.siglip_processor = None
        self.dinov2_encoder = None
        self.dinov2_processor = None
        
    def load_encoders(self, siglip_path: str, dinov2_path: str, cache_dir: Optional[Path] = None):
        """Load both image encoders"""
        logger.info(f"Loading SigLIP encoder: {siglip_path}")
        self.siglip_encoder = SiglipVisionModel.from_pretrained(
            siglip_path, 
            cache_dir=cache_dir,
            torch_dtype=self.dtype
        ).to(self.device)
        self.siglip_processor = SiglipImageProcessor.from_pretrained(
            siglip_path,
            cache_dir=cache_dir
        )
        
        logger.info(f"Loading DINOv2 encoder: {dinov2_path}")  
        self.dinov2_encoder = AutoModel.from_pretrained(
            dinov2_path,
            cache_dir=cache_dir,
            torch_dtype=self.dtype
        ).to(self.device)
        self.dinov2_processor = AutoImageProcessor.from_pretrained(
            dinov2_path,
            cache_dir=cache_dir
        )
        # Configure DINOv2 processor
        self.dinov2_processor.crop_size = dict(height=384, width=384)
        self.dinov2_processor.size = dict(shortest_edge=384)
        
        self.siglip_encoder.eval()
        self.dinov2_encoder.eval()
        
    def encode_siglip_image(self, siglip_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image with SigLIP"""
        siglip_image = siglip_image.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            res = self.siglip_encoder(siglip_image, output_hidden_states=True)
            siglip_image_embeds = res.last_hidden_state
            siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        return siglip_image_embeds, siglip_image_shallow_embeds
        
    def encode_dinov2_image(self, dinov2_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image with DINOv2"""
        dinov2_image = dinov2_image.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            res = self.dinov2_encoder(dinov2_image, output_hidden_states=True)
            dinov2_image_embeds = res.last_hidden_state[:, 1:]  # remove [CLS] token
            dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        return dinov2_image_embeds, dinov2_image_shallow_embeds


class InstantCharacterIPAdapter:
    """IP-Adapter for InstantCharacter"""
    
    def __init__(self, transformer, text_encoder_2, device: torch.device, dtype: torch.dtype):
        self.transformer = transformer
        self.text_encoder_2 = text_encoder_2
        self.device = device
        self.dtype = dtype
        self.image_proj_model = None
        
    def load_ip_adapter(self, ip_adapter_path: str, nb_token: int = 1024):
        """Load IP-Adapter weights and initialize processors for InvokeAI extension system"""
        logger.info(f"Loading IP-Adapter: {ip_adapter_path}")
        
        state_dict = torch.load(ip_adapter_path, map_location="cpu")
        
        # InvokeAI uses a custom extension system instead of attn_processors
        # We'll load the IP-Adapter weights and store them for use in the extension
        logger.info("Loading IP-Adapter weights for InvokeAI extension system")
        
        # Store IP-Adapter layers for use in the extension
        # These will be accessed by the InstantCharacterFluxExtension
        self.ip_adapter_layers = {}
        
        try:
            # Load IP-Adapter attention layers
            # We need to create the FluxIPAttnProcessor layers manually
            # since InvokeAI doesn't use the standard attn_processors system
            
            # Get transformer config to determine layer structure
            if hasattr(self.transformer, 'config'):
                config = self.transformer.config
                hidden_size = config.attention_head_dim * config.num_attention_heads
                ip_hidden_states_dim = self.text_encoder_2.config.d_model
            else:
                # Fallback dimensions based on FLUX architecture
                hidden_size = 3072  # Standard FLUX hidden size
                ip_hidden_states_dim = 4096  # T5 hidden size
                
            # Create IP-Adapter layers that will be used in the extension
            # We'll create a representative set of layers matching the state dict
            ip_adapter_state = state_dict["ip_adapter"]
            
            # Count the number of attention processors in the state dict
            num_layers = len([k for k in ip_adapter_state.keys() if k.endswith('.norm_ip_q.weight')])
            logger.info(f"Found {num_layers} IP-Adapter attention layers")
            
            # Create IP-Adapter layers
            for i in range(num_layers):
                layer_name = f"layer_{i}"
                layer = FluxIPAttnProcessor(
                    hidden_size=hidden_size,
                    ip_hidden_states_dim=ip_hidden_states_dim,
                ).to(self.device, dtype=self.dtype)
                self.ip_adapter_layers[layer_name] = layer
                
            # Load weights into the IP-Adapter layers
            ip_layers_module = torch.nn.ModuleList(self.ip_adapter_layers.values())
            missing_keys = ip_layers_module.load_state_dict(ip_adapter_state, strict=False)
            logger.info(f"Loaded IP-Adapter layers, missing keys: {missing_keys}")
            
        except Exception as e:
            logger.error(f"Failed to load IP-Adapter weights: {e}")
            logger.error(f"Transformer type: {type(self.transformer)}")
            logger.error(f"Available attributes: {[attr for attr in dir(self.transformer) if not attr.startswith('_')]}")
            raise
        
        # Initialize projection model
        logger.info("Initializing projection model")
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
        
        self.image_proj_model.eval()
        self.image_proj_model.to(self.device, dtype=self.dtype)
        
        # Load projection weights
        missing_keys = self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        logger.info(f"Loaded projection model, missing keys: {missing_keys}")
        
    def project_subject_embeddings(self, image_embeds_dict: Dict[str, torch.Tensor], timestep: torch.Tensor) -> torch.Tensor:
        """Project subject image embeddings"""
        if self.image_proj_model is None:
            raise RuntimeError("IP-Adapter not loaded")
            
        with torch.inference_mode():
            subject_embeds = self.image_proj_model(
                image_embeds_dict,
                timestep=timestep
            )
        return subject_embeds