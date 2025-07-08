"""
Bridge between InvokeAI and InstantCharacter systems
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
from einops import rearrange

from invokeai.app.invocations.baseinvocation import InvocationContext
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.flux.sampling_utils import get_schedule
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.sampling_utils import pack, unpack
from invokeai.backend.flux.text_conditioning import FluxTextConditioning
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState

from .instant_character_models import InstantCharacterImageEncoder, InstantCharacterIPAdapter
from .instant_character_flux_extension import create_instant_character_extensions

logger = InvokeAILogger.get_logger(__name__)



class InvokeAIInstantCharacterBridge:
    """Bridge between InvokeAI FLUX components and InstantCharacter logic"""
    
    def __init__(
        self, 
        transformer,
        vae, 
        text_encoder,
        text_encoder_2,
        clip_tokenizer,
        t5_tokenizer,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.clip_tokenizer = clip_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.device = device
        self.dtype = dtype
        
        # InstantCharacter components
        self.image_encoder = None
        self.ip_adapter = None
        
    def init_instant_character(
        self,
        context: InvocationContext,
        siglip_path: str = "google/siglip-so400m-patch14-384",
        dinov2_path: str = "facebook/dinov2-giant", 
        ip_adapter_path: str = "https://huggingface.co/Tencent/InstantCharacter/resolve/main/instantcharacter_ip-adapter.bin",
        nb_token: int = 1024
    ):
        """Initialize InstantCharacter components"""
        cache_dir = Path(context.config.get().download_cache_dir).resolve()
        
        # Initialize image encoders
        self.image_encoder = InstantCharacterImageEncoder(self.device, self.dtype)
        
        # Load image encoders through InvokeAI remote model system
        def load_siglip(model_path):
            from transformers import SiglipVisionModel, SiglipImageProcessor
            return (
                SiglipVisionModel.from_pretrained(model_path, torch_dtype=self.dtype),
                SiglipImageProcessor.from_pretrained(model_path)
            )
            
        def load_dinov2(model_path):
            from transformers import AutoModel, AutoImageProcessor
            encoder = AutoModel.from_pretrained(model_path, torch_dtype=self.dtype)
            processor = AutoImageProcessor.from_pretrained(model_path)
            processor.crop_size = dict(height=384, width=384)
            processor.size = dict(shortest_edge=384)
            return encoder, processor
            
        # Load IP-Adapter through InvokeAI system
        def load_ip_adapter(model_path):
            ip_path = Path(model_path)
            if ip_path.is_file():
                return str(ip_path)
            else:
                raise FileNotFoundError(f"IP-Adapter file not found: {model_path}")
        
        # Store model paths for lazy loading during image encoding
        self.siglip_path = siglip_path
        self.dinov2_path = dinov2_path
        self.load_siglip = load_siglip
        self.load_dinov2 = load_dinov2
        
        # Load IP-Adapter through InvokeAI system
        # Note: Using direct URL to bypass InvokeAI's file filtering
        with context.models.load_remote_model(source=ip_adapter_path, loader=load_ip_adapter) as ip_path:
            # Initialize IP-Adapter with downloaded file
            self.ip_adapter = InstantCharacterIPAdapter(
                self.transformer, 
                self.text_encoder_2,
                self.device,
                self.dtype
            )
            self.ip_adapter.load_ip_adapter(str(ip_path), nb_token)
            
        logger.info("InstantCharacter bridge initialized successfully")
    
    def encode_subject_image(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Encode subject image using both SigLIP and DINOv2"""
        if self.image_encoder is None:
            raise RuntimeError("Image encoder not initialized")
            
        # Prepare multi-resolution images like in original
        object_image_pil_low_res = [image.resize((384, 384))]
        object_image_pil_high_res = image.resize((768, 768))
        object_image_pil_high_res = [
            object_image_pil_high_res.crop((0, 0, 384, 384)),
            object_image_pil_high_res.crop((384, 0, 768, 384)),
            object_image_pil_high_res.crop((0, 384, 384, 768)),
            object_image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res)
        
        # Encode low resolution
        siglip_low_res = self.image_encoder.siglip_processor(
            images=object_image_pil_low_res, 
            return_tensors="pt"
        ).pixel_values
        siglip_embeds_low = self.image_encoder.encode_siglip_image(siglip_low_res)
        
        dinov2_low_res = self.image_encoder.dinov2_processor(
            images=object_image_pil_low_res,
            return_tensors="pt"
        ).pixel_values
        dinov2_embeds_low = self.image_encoder.encode_dinov2_image(dinov2_low_res)
        
        # Combine low resolution embeddings
        image_embeds_low_res_deep = torch.cat([siglip_embeds_low[0], dinov2_embeds_low[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_embeds_low[1], dinov2_embeds_low[1]], dim=2)
        
        # Encode high resolution
        siglip_high_res = self.image_encoder.siglip_processor(
            images=object_image_pil_high_res,
            return_tensors="pt"
        ).pixel_values
        siglip_high_res = siglip_high_res[None]
        siglip_high_res = rearrange(siglip_high_res, 'b n c h w -> (b n) c h w')
        siglip_embeds_high = self.image_encoder.encode_siglip_image(siglip_high_res)
        siglip_high_res_deep = rearrange(siglip_embeds_high[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        
        dinov2_high_res = self.image_encoder.dinov2_processor(
            images=object_image_pil_high_res,
            return_tensors="pt"
        ).pixel_values
        dinov2_high_res = dinov2_high_res[None]
        dinov2_high_res = rearrange(dinov2_high_res, 'b n c h w -> (b n) c h w')
        dinov2_embeds_high = self.image_encoder.encode_dinov2_image(dinov2_high_res)
        dinov2_high_res_deep = rearrange(dinov2_embeds_high[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        
        # Combine high resolution
        image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)
        
        return {
            "image_embeds_low_res_shallow": image_embeds_low_res_shallow,
            "image_embeds_low_res_deep": image_embeds_low_res_deep,
            "image_embeds_high_res_deep": image_embeds_high_res_deep,
        }
    
    def encode_prompts(self, prompt: str, negative_prompt: str = "") -> Dict[str, torch.Tensor]:
        """Encode text prompts using CLIP and T5"""
        # CLIP encoding
        clip_inputs = self.clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            clip_outputs = self.text_encoder(clip_inputs.input_ids)
            # Get pooled output for CLIP
            if hasattr(clip_outputs, 'pooler_output') and clip_outputs.pooler_output is not None:
                clip_embeds = clip_outputs.pooler_output.to(self.dtype)
            elif hasattr(clip_outputs, 'last_hidden_state'):
                # Use [CLS] token (first token) if no pooled output
                clip_embeds = clip_outputs.last_hidden_state[:, 0].to(self.dtype)
            else:
                # Fallback: mean pooling
                clip_embeds = clip_outputs[0].mean(dim=1).to(self.dtype)
            
        # T5 encoding  
        t5_inputs = self.t5_tokenizer(
            prompt,
            padding="max_length", 
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            t5_outputs = self.text_encoder_2(t5_inputs.input_ids)
            # Get hidden states for T5
            if hasattr(t5_outputs, 'last_hidden_state'):
                t5_embeds = t5_outputs.last_hidden_state.to(self.dtype)
            else:
                t5_embeds = t5_outputs[0].to(self.dtype)
            
        return {
            "clip_embeds": clip_embeds,
            "t5_embeds": t5_embeds,
        }
    
    def denoise_with_instant_character(
        self,
        prompt: str,
        subject_image: Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        subject_scale: float = 0.9,
        seed: int = 0,
    ) -> Image.Image:
        """Main denoising process using InvokeAI FLUX denoising with InstantCharacter IP-Adapter"""
        
        if self.ip_adapter is None:
            raise RuntimeError("IP-Adapter not initialized")
            
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Encode prompts
        prompt_embeds = self.encode_prompts(prompt)
        clip_embeds = prompt_embeds["clip_embeds"]
        t5_embeds = prompt_embeds["t5_embeds"]
        
        # Create text conditioning
        pos_conditioning = FluxTextConditioning(
            t5_embeddings=t5_embeds,
            clip_embeddings=clip_embeds,
            mask=None
        )
        
        # Prepare latents
        latent_channels = 16
        latents = torch.randn(
            (1, latent_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Pack latents for FLUX (reshape for transformer)
        packed_latents = pack(latents)
        
        # Create regional prompting extension
        pos_regional_ext = RegionalPromptingExtension.from_text_conditioning(
            text_conditioning=[pos_conditioning],
            redux_conditioning=[],
            img_seq_len=packed_latents.shape[1]
        )
        
        # TEMPORARY: Skip subject image encoding to test memory
        # subject_embeds_dict = self.encode_subject_image(subject_image)
        subject_embeds_dict = {
            "image_embeds_low_res_shallow": torch.zeros((1, 100, 2688), device=self.device, dtype=self.dtype),
            "image_embeds_low_res_deep": torch.zeros((1, 577, 2688), device=self.device, dtype=self.dtype),
            "image_embeds_high_res_deep": torch.zeros((1, 2308, 2688), device=self.device, dtype=self.dtype),
        }
        
        # Create position IDs for packed latents
        img_ids = torch.zeros(
            (1, packed_latents.shape[1], 3),
            device=self.device,
            dtype=torch.int32
        )
        
        # Get InvokeAI timestep schedule
        timesteps = get_schedule(
            num_steps=num_inference_steps,
            image_seq_len=packed_latents.shape[1],
        )
        
        # Extract actual timestep values for precomputation
        if hasattr(timesteps, 'cpu'):
            timestep_values = timesteps.cpu().numpy().tolist()
        elif isinstance(timesteps, (list, tuple)):
            timestep_values = list(timesteps)
        else:
            timestep_values = [float(t) for t in timesteps]
        
        # Step callback for progress
        def step_callback(state: PipelineIntermediateState) -> None:
            logger.debug(f"Step {state.step}/{len(timesteps)-1}, timestep: {state.timestep}")
        
        # Create InstantCharacter extensions with proper parameters
        pos_ip_adapter_extensions, neg_ip_adapter_extensions = create_instant_character_extensions(
            ip_adapter_layers=self.ip_adapter.ip_adapter_layers,
            image_proj_model=self.ip_adapter.image_proj_model,
            subject_embeds_dict=subject_embeds_dict,
            timesteps=timestep_values,
            weight=subject_scale,
            begin_step_percent=0.0,
            end_step_percent=1.0,
            device=self.device,
            dtype=self.dtype
        )
        
        # Run InvokeAI FLUX denoising
        denoised_latents = denoise(
            model=self.transformer,
            img=packed_latents,
            img_ids=img_ids,
            pos_regional_prompting_extension=pos_regional_ext,
            neg_regional_prompting_extension=None,
            timesteps=timesteps,
            step_callback=step_callback,
            guidance=guidance_scale,
            cfg_scale=[1.0] * len(timesteps),  # No CFG for now
            inpaint_extension=None,
            controlnet_extensions=[],
            pos_ip_adapter_extensions=pos_ip_adapter_extensions,
            neg_ip_adapter_extensions=neg_ip_adapter_extensions,
            img_cond=None,
            img_cond_seq=None,
            img_cond_seq_ids=None
        )
        
        # Unpack latents back to standard format
        unpacked_latents = unpack(denoised_latents, height // 8, width // 8)
        
        # Decode latents
        with torch.no_grad():
            images = self.vae.decode(unpacked_latents)
            
        # Convert to PIL
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = (images * 255).round().astype(np.uint8)
        
        return Image.fromarray(images[0])