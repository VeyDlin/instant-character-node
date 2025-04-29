import os
import torch
from .InstantCharacter.pipeline import InstantCharacterFluxPipeline

from typing import Optional
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
    ImageOutput
)

from invokeai.app.invocations.model import ModelIdentifierField, LoRAField


@invocation(
    "instant_character",
    title="Instant Character",
    tags=["generate", "character", "flux"],
    category="generate",
    version="1.0.0",
)
class InstantCharacterIvocation(BaseInvocation):
    """Instant Character"""
    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        ui_type=UIType.FluxMainModel,
        input=Input.Direct,
    )
    ip_adapter: str = InputField(default="Tencent/InstantCharacter")
    image_encoder: str = InputField(default="google/siglip-so400m-patch14-384")
    image_encoder_2: str = InputField(default="facebook/dinov2-giant")
    cpu_offload: bool = InputField(default=False)
    lora: Optional[LoRAField]  = InputField(
        default=None, description=FieldDescriptions.lora_model, title="LoRA"
    )
    subject_image: ImageField = InputField()
    prompt: str = InputField()
    prompt_with_lora_trigger: str = InputField()
    subject_scale: float = InputField(default=0.9, ge=0, le=2)      
    width: int = InputField(default=1024, multiple_of=64)             
    height: int = InputField(default=1024, multiple_of=64)             
    guidance_scale: float = InputField(default=3.5, ge=0, le=10) 
    num_inference_steps: int = InputField(default=28, ge=1, le=100) 
    seed: int = InputField(default=0, ge=0, le=0xffffffffffffffff)  

    def invoke(self, context: InvocationContext) -> ImageOutput:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_config = context.models.get_config(self.model)
        model_path = os.path.abspath(os.path.join(context.config.get().models_dir, model_config.path))
        pipe = InstantCharacterFluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        
        if self.cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)
        
        pipe.init_adapter(
            image_encoder_path=self.image_encoder,
            #cache_dir=image_encoder_cache_dir,
            image_encoder_2_path=self.image_encoder_2,
            #cache_dir_2=image_encoder_2_cache_dir,
            subject_ipadapter_cfg=dict(
                subject_ip_adapter_path=self.ip_adapter,
                nb_token=1024
            ),
        )

        subject_image_pil = context.images.get_pil(self.subject_image.image_name).convert('RGB')

        output = None
        if self.lora is None:
            output = pipe(
                prompt=self.prompt,
                height=self.height,
                width=self.width,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(self.seed),
                subject_image=subject_image_pil,
                subject_scale=self.subject_scale,
            )
        else:
            lora_config = context.models.get_config(self.lora.lora)
            lora_path = os.path.abspath(os.path.join(context.config.get().models_dir, lora_config.path))
            output = pipe.with_style_lora(
                lora_file_path=lora_path,
                lora_weight=self.lora.weight,
                prompt_with_lora_trigger=self.prompt_with_lora_trigger,

                prompt=self.prompt,
                height=self.height,
                width=self.width,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(self.seed),
                subject_image=subject_image_pil,
                subject_scale=self.subject_scale,
            )

        image_dto = context.images.save(image=output.images[0])

        return ImageOutput.build(image_dto)

