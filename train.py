import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils import save_checkpoint, plot_image



def train_fn(model, dataloader, optimizer, loss_fn, beta=1, scheduler=None, sampling=True, num_epochs=60, device="cuda"):
    """
    Performs a training with model trying to learn on dataloader.
    """

    time = datetime.now().strftime("%Y%m%d%H%M")
    train_loss = 0.0
    loss_history = []
    learning_rates = []

    store = []

    # Put model into training mode
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch} / {num_epochs}\n-------")

        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            # Put data and model on target device
            images = images.to(device)
            model = model.to(device)

            # Forward pass
            mu_z, log_var_z, mu_x = model(images)

            # Calculate loss per batch
            loss = loss_fn(images, mu_z, log_var_z, mu_x, beta=beta, device=device)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        store.append((mu_z, log_var_z))
        # Learning rate scheduler step
        if scheduler is not None:
            learning_rates.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Calculate and save average loss
        train_loss /= len(dataloader)
        loss_history.append(train_loss)
        print(f"Train loss: {train_loss:.4f}\n")
    
        # Sampling procedure
        if sampling:
            image_channels = images.shape[1]
            image_size = images.shape[2]

            # Original image
            image = images[0]

            # Reconstructed image
            _, _, mu_recon = model(image.unsqueeze(0))
            recon = model.decoder.sample(mu_recon).view(image_channels, image_size, image_size)   

            # Random generated image
            noise = torch.rand((1, image_channels, image_size, image_size)).to(device)
            _, _, mu_gen = model(noise)
            gen = model.decoder.sample(mu_gen).view(image_channels, image_size, image_size)

            if image_channels == 1:
                plot_image(image, recon, gen, cmap="gray")
            else:
                plot_image(image, recon, gen)


    # Saving model and optimizer state.
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()}

    save_checkpoint(checkpoint, filename=f"checkpoint_{time}.pth.tar")

    return loss_history, learning_rates, store