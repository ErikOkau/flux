import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")


# 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

# Make Nephis from Shadow Slave light novel in a high quality anime character. Appearance: Hair: Nephis has platinum blonde hair, which is one of her most distinguishing features. Her hair is usually described as straight and fine, often cascading down her back. The almost silvery hue of her hair gives her an ethereal and somewhat otherworldly appearance.  Eyes: Her eyes are described as a cold, piercing gray, almost like steel. They are sharp and convey a sense of intensity and focus. Her gaze is often described as unsettling, capable of seeing through others and revealing little about her own thoughts or emotions.  Skin: Nephis has pale, flawless skin, which complements her platinum hair. Her complexion adds to her ethereal and almost ghostly presence, making her stand out in any crowd.  Build: Nephis is physically fit and athletic, with a slender yet strong build. Her body reflects the rigorous training and harsh conditions she has endured. She moves with a grace that suggests both power and precision.  Clothing: Nephis often wears practical, functional clothing suited for combat and survival, yet there's a certain elegance to how she carries herself, even in the most utilitarian of outfits. Her clothing is usually dark or muted in color, helping her blend into her surroundings when necessary. 