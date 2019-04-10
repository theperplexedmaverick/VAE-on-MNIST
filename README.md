# VAE-on-MNIST
## Simple implementation of VAE
## 1 Algorithm and Python Code
Here we implement a simple Variational Autoencoder in MNIST. A VAE differs from a normal Autoencoder
in the sense that in a VAE, we force our network to generate an object following a Gaussian distribution, so
that the distribution becomes tractable. A vector of means and a vector standard deviations are generated
constituting the latent or the hidden vector in the bottleneck layer and they are then sampled and fed to
the decoder output. Since the distribution is already known, the decoder will return us completely new
objects that appear just like the objects our network has been trained with.
### 1.1 Dataset Initialization
We use MNIST dataset taht comprises of 28X28 images with one color channel. The model takes x-in as
inputs and produces Y as output which should be same to X-in. We use Leaky RelU for the non-linear
activation and keep-prob ensures regularization.
### 1.2 Building the model
Since we are using images, convolutional layers are appropriate for the encoder and the decoder neural
networks. We use 3 convolutional layers of 64 filters of (4X4) with strides of 2 and 1. We compute the
z-values by computing the sum of the mean vector and the standard deviation vector multiplied by a vector
of normalized values. The same model is replicated for the decoder as well that tries to take the z values
and reconstructs the image.
### 1.3 Optimizer and Loss
For computing the image reconstruction loss, we simply use squared difference (which could lead to images
sometimes looking a bit fuzzy). This loss is combined with the Kullback-Leibler divergence(that calculates
the divergence between 2 distributions), which makes sure our latent values will be sampled from a normal
distribution. Then we use and Adam optimizer to optimize the loss.
### 1.4 Operation
We run sessions for 30000 iterations in batches of 64. Every 3000 iterations, we print the loss and the
set of input and the decoded image. Finally, we create new images with a normally distributed randomly
generated dataset. This time, we don’t dropout by keeping the keep-prob=1.
## 2 Training and Testing Performance
We start with a loss of 176.69328 and by the last iteration we reduce the loss to 27.416351.
0 176.69328 176.68802 0.005269118
3000 30.50178 21.248009 9.253771
6000 29.328245 19.411682 9.916564
9000 27.120705 17.117283 10.003421
12000 28.194715 16.910442 11.284273
15000 29.4137 19.280684 10.133017
18000 27.697973 17.263563 10.43441
21000 26.426815 16.04147 10.3853445
24000 27.906342 16.722023 11.184317
27000 27.416351 16.283142 11.133209

We then create new objects using our model:-
## 3 References

• https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-
978675c95776

• https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
