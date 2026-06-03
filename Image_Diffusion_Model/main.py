from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Get prompt from user
prompt = input("Enter your image description: ")

# Generate image
with torch.autocast(device):
    image = pipe(prompt).images[0]

# Display image
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()

# Save image
filename = "generated_image.png"
image.save(filename)

print(f"Image saved as {filename}")