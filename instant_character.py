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

from .instant_character_bridge import InvokeAIInstantCharacterBridge

logger = InvokeAILogger.get_logger(__name__)


@invocation(
    "instant_character_flux",
    title="InstantCharacter - FLUX",
    tags=["flux", "character", "ip-adapter"],
    category="flux",
    version="1.0.0",
)
class InstantCharacterFluxInvocation(BaseInvocation):
    """Apply InstantCharacter IP-Adapter to FLUX models for character-consistent generation"""
    
    # Model inputs from previous nodes (FluxModelLoader, FluxLoRALoader)
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description="FLUX transformer from FluxModelLoader",
        input=Input.Connection,
        title="Transformer"
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description="CLIP from FluxModelLoader", 
        input=Input.Connection,
        title="CLIP"
    )
    t5_encoder: Optional[T5EncoderField] = InputField(
        default=None,
        description="T5 Encoder from FluxModelLoader",
        input=Input.Connection,
        title="T5 Encoder"
    )
    vae: Optional[VAEField] = InputField(
        default=None,
        description="VAE from FluxModelLoader",
        input=Input.Connection,
        title="VAE"
    )
    
    # InstantCharacter specific parameters
    subject_image: ImageField = InputField(
        description="Reference character image",
        title="Character Image"
    )
    prompt: str = InputField(
        description="Text prompt for generation",
        title="Prompt"
    )
    
    # InstantCharacter model paths
    ip_adapter: str = InputField(
        default="https://huggingface.co/Tencent/InstantCharacter/resolve/main/instantcharacter_ip-adapter.bin",
        description="InstantCharacter IP-Adapter direct file URL",
        title="IP-Adapter Model"
    )
    image_encoder: str = InputField(
        default="google/siglip-so400m-patch14-384",
        description="SigLIP image encoder model path",
        title="SigLIP Encoder"
    )
    image_encoder_2: str = InputField(
        default="facebook/dinov2-giant",
        description="DINOv2 image encoder model path", 
        title="DINOv2 Encoder"
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
    guidance_scale: float = InputField(
        default=3.5,
        ge=0.0,
        le=10.0,
        description="Guidance scale for generation",
        title="Guidance Scale"
    )
    num_inference_steps: int = InputField(
        default=28,
        ge=1,
        le=100,
        description="Number of denoising steps",
        title="Steps"
    )
    seed: int = InputField(
        default=0,
        ge=0,
        le=0xffffffffffffffff,
        description="Random seed for generation",
        title="Seed"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Validate inputs
        if not all([self.transformer, self.clip, self.t5_encoder, self.vae]):
            raise ValueError(
                "InstantCharacter requires all model components. "
                "Please connect outputs from 'Main Model - FLUX' node."
            )
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inference_dtype = torch.bfloat16
        
        logger.info("Starting InstantCharacter FLUX generation")
        
        # Load model components using the field references
        load_clip_tokenizer = self.clip.tokenizer
        load_clip_text_encoder = self.clip.text_encoder
        load_t5_encoder_tokenizer = self.t5_encoder.tokenizer
        load_t5_encoder_text_encoder = self.t5_encoder.text_encoder
        load_transformer = self.transformer.transformer
        load_vae = self.vae.vae
        
        # Load all components with InvokeAI memory management
        with (
            context.models.load(load_clip_text_encoder).model_on_device() as (_, clip_text_encoder),
            context.models.load(load_clip_tokenizer) as clip_tokenizer,
            context.models.load(load_t5_encoder_text_encoder).model_on_device() as (_, t5_text_encoder),
            context.models.load(load_t5_encoder_tokenizer) as t5_tokenizer,
            context.models.load(load_vae).model_on_device() as (_, vae_model),
            context.models.load(load_transformer).model_on_device() as (_, transformer),
        ):
            logger.info("All FLUX models loaded successfully")
            
            # Create bridge between InvokeAI and InstantCharacter
            bridge = InvokeAIInstantCharacterBridge(
                transformer=transformer,
                vae=vae_model,
                text_encoder=clip_text_encoder,
                text_encoder_2=t5_text_encoder,
                clip_tokenizer=clip_tokenizer,
                t5_tokenizer=t5_tokenizer,
                device=device,
                dtype=inference_dtype
            )
            
            # Initialize InstantCharacter components
            bridge.init_instant_character(
                context=context,
                siglip_path=self.image_encoder,
                dinov2_path=self.image_encoder_2,
                ip_adapter_path=self.ip_adapter,
                nb_token=1024
            )
            
            # Get subject image
            subject_image_pil = context.images.get_pil(self.subject_image.image_name).convert('RGB')
            
            # Run InstantCharacter generation
            output_image = bridge.denoise_with_instant_character(
                prompt=self.prompt,
                subject_image=subject_image_pil,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                subject_scale=self.subject_scale,
                seed=self.seed
            )
            
            # Save result
            image_dto = context.images.save(image=output_image)
            
            logger.info("InstantCharacter generation completed successfully")
            return ImageOutput.build(image_dto)