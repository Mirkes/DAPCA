The data is taken and decoded from svmlight format to .mat from
https://github.com/GRAAL-Research/domain_adversarial_neural_network

.mat files contain boths data matrix ('data') and the label vector ('label')

The [Ganin et al, 2016](https://jmlr.org/papers/volume17/15-239/15-239.pdf) study suggests the following 12 domain adaptation problems
with following accuracies achieved (for the original data - first three numerical columns, and for mSAD-transformed data - last three numerical columns):

| Source      | Target      | DANN | NN   | SVM  |  DANN | NN   | SVM  |
|-------------|-------------|------|------|------|-------|------|------|
| books       | dvd         | .784 | .790 | .799 |  .829 | .824 | .830 |
| books       | electronics | .733 | .747 | .748 |  .804 | .770 | .766 |
| books       | kitchen     | .779 | .778 | .769 |  .843 | .842 | .821 |
| dvd         | books       | .723 | .720 | .743 |  .825 | .823 | .826 |
| dvd         | electronics | .754 | .732 | .748 |  .809 | .768 | .739 |
| dvd         | kitchen     | .783 | .778 | .746 |  .849 | .853 | .842 |
| electronics | books       | .713 | .709 | .705 |  .774 | .770 | .762 |
| electronics | dvd         | .738 | .733 | .726 |  .781 | .759 | .770 |
| electronics | kitchen     | .854 | .854 | .847 |  .881 | .863 | .847 |
| kitchen     | books       | .709 | .708 | .707 |  .718 | .721 | .769 |
| kitchen     | dvd         | .740 | .739 | .736 |  .789 | .789 | .788 |
| kitchen     | electronics | .843 | .841 | .842 |  .856 | .850 | .861 |

The domain adaptation problem should be trained using the pair of training sets (2000 samples), and tested using test dataset parts (from 2000 to 6000 samples).

The datasets contain 5000 features ordered by frequency such that only first k of them could be used for training.