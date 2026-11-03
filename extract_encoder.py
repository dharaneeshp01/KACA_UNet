import torch

# Load your successful 9-class model
checkpoint = torch.load('./out_seg/best_model.pth', map_location='cpu')
unet_state = checkpoint['model']

encoder_state = {}
for key, value in unet_state.items():
    if key.startswith('encoder.'):
        # Remove the 'encoder.' prefix so the names match exactly for load_state_dict
        clean_key = key.replace('encoder.', '', 1)
        encoder_state[clean_key] = value

# Save the perfectly formatted "eyes"
torch.save(encoder_state, './out_seg/pretrained_encoder.pth')
print(f"Successfully extracted {len(encoder_state)} encoder layers!")
