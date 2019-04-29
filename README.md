## Autoencoder-Networks-of-my-Masterthesis
This repo contains a summary of the most important codes of my master thesis. A detailed discussion of the models can be found in my master thesis (see pdf-file).

### GS-VAE
The GS-VAE is capable to autoencode spectrogram snippets of fixed length. The latent space is discrete, using the Concrete relaxation. 

### Recurrent GS-VAE
The recurrent GS-VAE is capable to autoencode spectrogram snippets of variable length. The latent space is discrete, using the Concrete relaxation. 

### Recurrent VAE-GAN
The recurrent VAE-GAN is capable to autoencode spectrogram snippets of variable length. The latent space is continuous. The loss function is a mixture of L2-loss and an adversarial discriminator. The probabilistic encoder learns a Gaussian approximative posterior p(z|x) and the decoder f(x|z) is deterministic. It is straight foward to turn the VAE in a completely deterministic autoencoder by learning only the means of the Gaussian. If one sets the argument "sample=True" of the forward-function in rnn_vs.py, the network chooses always the means of the Gaussian. Doing this during training is equivalent to train a deterministic autoencoder.

### Data loading stuff
The folder "data loading stuff" contains important functions to preprocess and load the data. However, it is very specific to the tasks in my master thesis and might not be interesting for the general reader.

