"""
Optimized InstantCharacter node for InvokeAI using proper IP-Adapter extension system
"""
from typing import Optional
from pathlib import Path
import torch
from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    ImageField,
    Input,
    UIType,
    FieldDescriptions,
    ModelIdentifierField,
    ImageOutput,
    BaseInvocationOutput,
    OutputField
)
from invokeai.app.invocations.model import (
    TransformerField, 
    CLIPField, 
    T5EncoderField, 
    VAEField,
    LoRAField
)
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.flux.sampling_utils import get_schedule, pack, unpack
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.text_conditioning import FluxTextConditioning
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState

from .instant_character_invoke_adapter import InstantCharacterInvokeAdapter
from .instant_character_invoke_extension import InstantCharacterModel
from .instant_character_models import InstantCharacterImageEncoder

logger = InvokeAILogger.get_logger(__name__)


@invocation(
    "instant_character_flux_optimized",
    title="InstantCharacter FLUX (Optimized)",
    tags=["flux", "character", "ip-adapter"],
    category="flux",
    version="2.0.0",
)
class InstantCharacterFluxOptimizedInvocation(BaseInvocation):
    """Optimized InstantCharacter for FLUX using InvokeAI's IP-Adapter extension system"""
    
    # Model inputs from FLUX workflow
    transformer: TransformerField = InputField(
        description="FLUX transformer from FluxModelLoader",
        input=Input.Connection,
        title="Transformer"
    )
    clip: CLIPField = InputField(
        description="CLIP from FluxModelLoader", 
        input=Input.Connection,
        title="CLIP"
    )
    t5_encoder: T5EncoderField = InputField(
        description="T5 Encoder from FluxModelLoader",
        input=Input.Connection,
        title="T5 Encoder"
    )
    vae: VAEField = InputField(
        description="VAE from FluxModelLoader",
        input=Input.Connection,
        title="VAE"
    )
    
    # InstantCharacter parameters
    subject_image: ImageField = InputField(
        description="Reference character image",
        title="Character Image"
    )
    prompt: str = InputField(
        description="Text prompt for generation",
        title="Prompt"
    )
    
    # Model paths
    ip_adapter_path: str = InputField(
        default="https://huggingface.co/Tencent/InstantCharacter/resolve/main/instantcharacter_ip-adapter.bin",
        description="InstantCharacter IP-Adapter file URL",
        title="IP-Adapter Model"
    )
    
    # Generation parameters
    subject_scale: float = InputField(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Strength of character conditioning",
        title="Character Strength"
    )
    width: int = InputField(
        default=1024,
        multiple_of=64,
        description="Image width",
        title="Width"
    )
    height: int = InputField(
        default=1024,
        multiple_of=64,
        description="Image height",
        title="Height"
    )
    num_inference_steps: int = InputField(
        default=28,
        ge=1,
        le=100,
        description="Number of denoising steps",
        title="Steps"
    )
    guidance_scale: float = InputField(
        default=3.5,
        ge=0.0,
        le=10.0,
        description="Guidance scale for generation",
        title="Guidance Scale"
    )
    seed: int = InputField(
        default=0,
        ge=0,
        le=0xffffffffffffffff,
        description="Random seed for generation",
        title="Seed"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        
        # Load all required models
        load_transformer = self.transformer.transformer
        load_vae = self.vae.vae
        load_clip_text_encoder = self.clip.text_encoder
        load_clip_tokenizer = self.clip.tokenizer
        load_t5_text_encoder = self.t5_encoder.text_encoder
        load_t5_tokenizer = self.t5_encoder.tokenizer
        
        with (
            context.models.load(load_clip_text_encoder).model_on_device() as (_, clip_text_encoder),
            context.models.load(load_clip_tokenizer) as clip_tokenizer,
            context.models.load(load_t5_text_encoder).model_on_device() as (_, t5_text_encoder),
            context.models.load(load_t5_tokenizer) as t5_tokenizer,
            context.models.load(load_vae).model_on_device() as (_, vae_model),
            context.models.load(load_transformer).model_on_device() as (_, transformer),
        ):
            # Load InstantCharacter model
            def load_ic_model(model_path):
                return InstantCharacterInvokeAdapter.load_model(
                    model_path=model_path,
                    device=device,
                    dtype=dtype
                )
            
            with context.models.load_remote_model(
                source=self.ip_adapter_path, 
                loader=load_ic_model
            ) as ic_model:
                
                # Load image encoders
                image_encoder = InstantCharacterImageEncoder(device, dtype)
                
                # Load SigLIP
                def load_siglip(model_path):
                    from transformers import SiglipVisionModel, SiglipImageProcessor
                    return (
                        SiglipVisionModel.from_pretrained(model_path, torch_dtype=dtype),
                        SiglipImageProcessor.from_pretrained(model_path)
                    )
                
                with context.models.load_remote_model(
                    source="google/siglip-so400m-patch14-384",
                    loader=load_siglip
                ) as (siglip_encoder, siglip_processor):
                    image_encoder.siglip_encoder = siglip_encoder.to(device)
                    image_encoder.siglip_processor = siglip_processor
                    
                    # Load DINOv2
                    def load_dinov2(model_path):
                        from transformers import AutoModel, AutoImageProcessor
                        encoder = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
                        processor = AutoImageProcessor.from_pretrained(model_path)
                        processor.crop_size = dict(height=384, width=384)
                        processor.size = dict(shortest_edge=384)
                        return encoder, processor
                        
                    with context.models.load_remote_model(
                        source="facebook/dinov2-giant",
                        loader=load_dinov2
                    ) as (dinov2_encoder, dinov2_processor):
                        image_encoder.dinov2_encoder = dinov2_encoder.to(device)
                        image_encoder.dinov2_processor = dinov2_processor
                        
                        # Get subject image
                        subject_image_pil = context.images.get_pil(self.subject_image.image_name).convert('RGB')
                        
                        # Encode subject image
                        subject_embeds_dict = InstantCharacterInvokeAdapter.encode_subject_image(
                            context=context,
                            image_encoder=image_encoder,
                            image=subject_image_pil,
                            device=device,
                            dtype=dtype
                        )
                        
                        # Encode text prompts
                        generator = torch.Generator(device=device).manual_seed(self.seed)
                        
                        # CLIP encoding
                        clip_inputs = clip_tokenizer(
                            self.prompt,
                            padding="max_length",
                            max_length=77,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        with torch.no_grad():
                            clip_outputs = clip_text_encoder(clip_inputs.input_ids)
                            if hasattr(clip_outputs, 'pooler_output') and clip_outputs.pooler_output is not None:
                                clip_embeds = clip_outputs.pooler_output.to(dtype)
                            else:
                                clip_embeds = clip_outputs.last_hidden_state[:, 0].to(dtype)
                        
                        # T5 encoding
                        t5_inputs = t5_tokenizer(
                            self.prompt,
                            padding="max_length",
                            max_length=512,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        with torch.no_grad():
                            t5_outputs = t5_text_encoder(t5_inputs.input_ids)
                            t5_embeds = t5_outputs.last_hidden_state.to(dtype)
                        
                        # Create text conditioning
                        pos_conditioning = FluxTextConditioning(
                            t5_embeddings=t5_embeds,
                            clip_embeddings=clip_embeds,
                            mask=None
                        )
                        
                        # Prepare latents
                        latent_channels = 16
                        latents = torch.randn(
                            (1, latent_channels, self.height // 8, self.width // 8),
                            generator=generator,
                            device=device,
                            dtype=dtype
                        )
                        
                        # Pack latents
                        packed_latents = pack(latents)
                        
                        # Create regional prompting extension
                        pos_regional_ext = RegionalPromptingExtension.from_text_conditioning(
                            text_conditioning=[pos_conditioning],
                            redux_conditioning=[],
                            img_seq_len=packed_latents.shape[1]
                        )
                        
                        # Position IDs
                        img_ids = torch.zeros(
                            (1, packed_latents.shape[1], 3),
                            device=device,
                            dtype=torch.int32
                        )
                        
                        # Get timesteps
                        timesteps = get_schedule(
                            num_steps=self.num_inference_steps,
                            image_seq_len=packed_latents.shape[1],
                        )
                        
                        # Extract timestep values
                        if hasattr(timesteps, 'cpu'):
                            timestep_values = timesteps.cpu().numpy().tolist()
                        else:
                            timestep_values = list(timesteps)
                        
                        # Create InstantCharacter extensions
                        pos_ic_extensions, neg_ic_extensions = InstantCharacterInvokeAdapter.create_extension(
                            model=ic_model,
                            subject_embeds_dict=subject_embeds_dict,
                            timesteps=timestep_values,
                            weight=self.subject_scale,
                            begin_step_percent=0.0,
                            end_step_percent=1.0,
                        )
                        
                        # Step callback
                        def step_callback(state: PipelineIntermediateState) -> None:
                            pass  # Could add progress reporting here
                        
                        # Denoise
                        denoised_latents = denoise(
                            model=transformer,
                            img=packed_latents,
                            img_ids=img_ids,
                            pos_regional_prompting_extension=pos_regional_ext,
                            neg_regional_prompting_extension=None,
                            timesteps=timesteps,
                            step_callback=step_callback,
                            guidance=self.guidance_scale,
                            cfg_scale=[1.0] * len(timesteps),
                            inpaint_extension=None,
                            controlnet_extensions=[],
                            pos_ip_adapter_extensions=pos_ic_extensions,
                            neg_ip_adapter_extensions=neg_ic_extensions,
                            img_cond=None,
                            img_cond_seq=None,
                            img_cond_seq_ids=None
                        )
                        
                        # Unpack and decode
                        unpacked_latents = unpack(denoised_latents, self.height // 8, self.width // 8)
                        
                        with torch.no_grad():
                            images = vae_model.decode(unpacked_latents)
                            
                        # Convert to PIL
                        images = (images / 2 + 0.5).clamp(0, 1)
                        images = images.permute(0, 2, 3, 1).cpu().numpy()
                        images = (images * 255).round().astype('uint8')
                        
                        import numpy as np
                        output_image = Image.fromarray(images[0])
                        
                        # Save
                        image_dto = context.images.save(image=output_image)
                        return ImageOutput.build(image_dto)