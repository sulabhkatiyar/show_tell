# show_tell
## Pytorch Implementation of Show and Tell: Neural Image Caption (NIC) Generator model

*Note:* This is a work in progress

In this work, I have re-implemented the paper: [Show and tell: A neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html). This work has been used as baseline in some of my research works where I have compared the performance of this method with other methods in the literature. Hence, to undertake a fair comparison, I have implemented this method with similar hyperparameter settings as other methods that have been studied in my works. 
Thus, there are some differences in this implementation as compared to the method used in the paper. These are as follows:
1. In the paper, authors use GoogLeNet CNN as encoder for image feature extraction but I have used VGG-16 as encoder.
1. I have not used model ensembles in this implementation. The authors determine that using model ensembles helps them enhance the performance by around 1-2 points on most BLEU metric.
1. I have not used batch normalization for inputs. 
1. Beam width of 3 has been used for inference. I have observed that, as compared to beam width of 20 which has been used in this paper, beam width of 3 provides better results.
