This dataset is prepared to benchmark the domain adaptation methods.

The source dataset is the standard MNIST dataset, expanded in 3 color channels (hence, containing 28x28x3 features, but the color data in all channels is repeated)

The target dataset is MNIST_M dataset obtained by blending MNIST data with colorful background taken from some other images. It is also prepared as a color image containing 3 channels.
Therefore MNIST_M represents a 'corrupted' version of MNIST.

The MNIST_M image data were downloaded from [http://yaroslav.ganin.net/](http://yaroslav.ganin.net/).
Each image was rescaled to 28x28 image (since the images were 32x32 size) and then encoded as a flat vector.

The MNIST_M_train.mat file is 220Mb size and should be downloaded [separately](https://drive.google.com/file/d/1rEDSvbeQ5XR2k3qgioXzTptd-IfDSlsp/view?usp=sharing).

.mat files contain boths data matrix ('data') and the label vector ('label')

The [Ganin et al, 2016](https://jmlr.org/papers/volume17/15-239/15-239.pdf) study 
reports the following accuracies achieved on training on MNIST data (source) and testing on MNIST_M data (target)
       	
| METHOD                      | ACCURACY(Gain)| 
|-----------------------------|---------------|
| Source only                 | .5225         | 
| SA (Fernando et al., 2013)  | .5690 (4.1%)  | 
| DANN                        | .766 (52.9%)  | 
| Train on target             | .9596         | 

Table legend: The first row corresponds to the lower performance bound
(i.e., if no adaptation is performed). The last row corresponds to training on
the target domain data with known class labels (upper bound on the DA performance).
For each of the two DA methods (ours and Fernando et al., 2013) we
show how much of the gap between the lower and the upper bounds was covered
(in brackets).

The following image depicts the difficulty of the domain adaptation problem: two datasets have very different axes of the main principal variance. Also, there seems to be a non-linear 'twist' in MNIST_M (probably, related to dark vs light backgrounds).

![PCA of test sets MNIST and MNIST_M](https://github.com/Mirkes/DAPCA/blob/main/images/MNIST_vs_MNIST_M_PCA.png) 