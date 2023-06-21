import torch
from torch import nn


class Encoder(nn.Module):
  def __init__(self, network):
    super(Encoder, self).__init__()

    # Initialization of the encoder network.
    self.encoder = network

  @staticmethod
  def reparameterization(mu, log_var):
    """
    This function reparameterizes nodes to prevent exploding gradients and provide backpropagation.
    The formula for reparameterization is the following: z = mu + std * epsilon,
    where epsilon ~ Normal(0, 1).
    """

    # Getting standard deviation from log-variance.
    std = torch.exp(0.5 * log_var)

    # Sampling epsilon from Normal(0, 1).
    eps = torch.randn_like(std)
    return mu + std * eps

  def encode(self, x):
    """
    This function returns the output of the encoder network, namely the means and standard deviations.
    """
    # Calculating the output of the encoder network of size 2L
    z = self.encoder(x)
    
    # Dividing the output to the mean and the log-variance
    mu, log_var = torch.chunk(z, 2, dim = 1)
    return mu, log_var

  def sample(self, mu, log_var):
    """
    Sampling procedure, mainly for forward pass and model testing.
    """

    z = self.reparameterization(mu, log_var)
    return z

  def forward(self, x):
    """
    PyTorch forward pass for nn.Module.
    """

    mu, log_var = self.encode(x)
    return self.sample(mu, log_var)
  

class Decoder(nn.Module):
  def __init__(self, network, num_vals):
    super(Decoder, self).__init__()

    # Initialization of the decoder network.
    self.decoder = network

    self.num_vals = num_vals

  def decode(self, z):
    """
    This function returns the output of the decoder network.
    """
    # Calculating the output of the decoder network.
    h = self.decoder(z)
    
    # Getting batch size and the dimensionality od x.
    b = h.shape[0]
    d = h.shape[2]

    h = h.view(b, d ** 2, self.num_vals)

    # To get probabilities for every pixel, we apply softmax function.
    mu = torch.softmax(h, 2)
    return mu

  def sample(self, mu):
    """
    Sampling procedure.
    """
    b = mu.shape[0]
    m = mu.shape[1]

    mu = mu.view(b, -1, self.num_vals)
    p = mu.view(-1, self.num_vals)

    x_new = torch.multinomial(p, num_samples=1).view(b, m)
    return x_new

  def forward(self, z):
    """
    PyTorch forward pass for nn.Module.
    """
    mu = self.decode(z)
    return self.sample(mu)
  

class VariationalAutoEncoder(nn.Module):
  def __init__(self, encoder_network, decoder_network, num_vals=256):
    super(VariationalAutoEncoder, self).__init__()

    # Initialization of the model components.
    self.encoder = Encoder(network=encoder_network)
    self.decoder = Decoder(network=decoder_network, num_vals=num_vals)

    self.num_vals = num_vals

  def forward(self, x):
    # encoder path
    mu_z, log_var_z = self.encoder.encode(x)
    z = self.encoder.sample(mu_z, log_var_z)

    # decoder path
    mu_x = self.decoder.decode(z)

    return mu_z, log_var_z, mu_x