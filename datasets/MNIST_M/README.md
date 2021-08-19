This dataset is prepared to benchmark the domain adaptation methods.

The source dataset is the standard MNIST dataset, expanded in 3 color channels (hence, containing 28*28*3 features, but the color data in all channels is repeated)

The target dataset is MNIST_M dataset obtained by blending MNIST data with colorful background taken from some other images. It is also prepared as a color image containing 3 channels.
Therefore MNIST_M represents a 'corrupted' version of MNIST.

The MNIST_M image data were downloaded from [http://yaroslav.ganin.net/](http://yaroslav.ganin.net/).
Each image was rescaled to 28*28 image (since the images were 32*32 size) and then encoded as a flat vector.

.mat files contain boths data matrix ('data') and the label vector ('label')