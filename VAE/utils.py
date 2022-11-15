from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as fn

def pil_to_tensor(img):
    """Converts the PIL img to a tensor usable for training"""
    img = ToTensor()(img)
    return img

def plot_latents(model, data, mode='ae'):
    """
    Encodes images into latent space and produce a plot
    using the latent embeddings of th MNIST dataset.
    NOTE: Make sure the latent embeddings has only 2 dims. 
    Args:
        data : dataloader for images, labels.
        
    Returns:
        Plot in latent space
    """
    latents = []
    labels = []
    for i, batch in enumerate(data):
        imgs, digits = batch 
        imgs = imgs.squeeze(1).reshape(-1, 28*28)
        if mode == 'ae':
            embeds, _ = model(imgs)
        if mode == 'vae':
            embeds, _,_,_ = model(imgs)
        embeds = embeds.detach().numpy().tolist()
        digits = digits.detach().numpy().tolist()
        latents.extend(embeds)
        labels.extend(digits)
    latents = np.asarray(latents)
    print(f"Labels:{len(labels)}, Data:{len(latents)}")
    plt.scatter(latents[:,0], latents[:,1], c=labels, cmap='tab10')
    plt.colorbar()
    
def generate_latents(model, images, mode='ae'):
    """
    Encodes images into latent space.
    Args:
        images : tensors of shape (-1, 1, 28, 28)
    Returns:
        Reconstructed img
    """
    imgs = images.squeeze(1).reshape(-1, 28*28)
    if mode == 'ae':
        latents, reconstructed = model(imgs)
    if mode == 'vae':
        latents, reconstructed,_,_ = model(imgs)
    
    return imgs.detach().numpy(), reconstructed.detach().numpy()
