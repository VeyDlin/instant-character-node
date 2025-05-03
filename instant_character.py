from pathlib import Path
import torch
from contextlib import ExitStack
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer, T5TokenizerFast
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
from invokeai.backend.flux.model import Flux
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.app.invocations.model import ModelIdentifierField, LoRAField
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)

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
    t5_encoder_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.t5_encoder, ui_type=UIType.T5EncoderModel, input=Input.Direct, title="T5 Encoder"
    )
    clip_embed_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.clip_embed_model,
        ui_type=UIType.CLIPEmbedModel,
        input=Input.Direct,
        title="CLIP Embed",
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
        inference_dtype = torch.bfloat16

        load_clip_tokenizer = self.clip_embed_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        load_clip_text_encoder = self.clip_embed_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        clip_text_encoder_info = context.models.load(load_clip_text_encoder)
        clip_text_encoder_config = clip_text_encoder_info.config
        assert clip_text_encoder_config is not None

        load_t5_encoder_tokenizer = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model)
        load_t5_encoder_text_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model)
        t5_encoder_info = context.models.load(load_t5_encoder_text_encoder)
        t5_encoder_config = t5_encoder_info.config
        assert t5_encoder_config is not None

        load_transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        transformer_config = context.models.get_config(load_transformer)
        assert transformer_config is not None
        
        with (
            clip_text_encoder_info.model_on_device() as (cached_weights, clip_text_encoder),
            context.models.load(load_clip_tokenizer) as clip_tokenizer,
            t5_encoder_info.model_on_device() as (cached_weights, t5_text_encoder),
            context.models.load(load_t5_encoder_tokenizer) as t5_tokenizer,
            ExitStack() as exit_stack,
        ):
            assert isinstance(clip_text_encoder, CLIPTextModel)
            assert isinstance(clip_tokenizer, CLIPTokenizer)
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(t5_tokenizer, (T5Tokenizer, T5TokenizerFast))

            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(load_transformer).model_on_device()
            )
            assert isinstance(transformer, Flux)
            if transformer_config.format in [ModelFormat.Checkpoint]:
                model_is_quantized = False
            elif transformer_config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported model format: {transformer_config.format}")

            pipe = InstantCharacterFluxPipeline(
                transformer=transformer,
                text_encoder=clip_text_encoder,
                tokenizer=clip_tokenizer,
                text_encoder_2=t5_text_encoder,
                tokenizer_2=t5_tokenizer,
                torch_dtype=inference_dtype,
            )
            
            if self.cpu_offload:
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.to(device)
            
            pipe.init_adapter(
                image_encoder_path=self.image_encoder,
                cache_dir=Path(context.config.get().download_cache_dir).resolve(),
                image_encoder_2_path=self.image_encoder_2,
                cache_dir_2=Path(context.config.get().download_cache_dir).resolve(),
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
                lora_path = (Path(context.config.get().models_dir) / lora_config.path).resolve()
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

