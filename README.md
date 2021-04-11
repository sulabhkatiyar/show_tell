# show_tell
## Pytorch Implementation of Show and Tell: Neural Image Caption (NIC) Generator model

*Note:* This is a work in progress

In this work, I have re-implemented the paper: [Show and tell: A neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html). This work has been used as baseline in some of my research works where I have compared the performance of this method with other methods in the literature. Hence, to undertake a fair comparison, I have implemented this method with similar hyperparameter settings as other methods that have been studied in my works. 
Thus, there are some differences in this implementation as compared to the method used in the paper. These are as follows:
1. In the paper, authors use GoogLeNet CNN as encoder for image feature extraction but I have used VGG-16 as encoder.
1. I have not used model ensembles in this implementation. The authors determine that using model ensembles helps them enhance the performance by around 1-2 points on most BLEU metric.
1. I have not used batch normalization for inputs. 
1. Beam width of 3 has been used for inference. I have observed that, as compared to beam width of 20 which has been used in this paper, beam width of 3 provides better results. Here, I have quoted the results with all beam sizes, for the sake of completeness.

**For Flickr30k dataset:**

|Result |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
|Paper | 20 | 0.663 | 0.423 | 0.277 | 0.183 | __ | __ | __ | __ |
|Our | 1 | 0.561 | 0.373 | 0.245 | 0.162 | 0.174 | 0.323 | 0.114 | 0.410 |
|Our | 3 | 0.600 | 0.400 | 0.266 | 0.177 | 0.170 | 0.354 | 0.113 | 0.415 |
|Our | 5 | 0.610 | 0.410 | 0.274 | 0.184 | 0.169 | 0.363 | 0.114 | 0.416 |
|Our | 10 | 0.600 | 0.401 | 0.267 | 0.179 | 0.163 | 0.350 | 0.108 | 0.409 |
|Our | 15 | 0.590 | 0.396 | 0.262 | 0.175 | 0.160 | 0.352 | 0.106 | 0.405 |
|Our | 20 | 0.581 | 0.390 | 0.257 | 0.170 | 0.157 | 0.347 | 0.103 | 0.401 |

**For MSCOCO dataset:**

|Result |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
|Paper |  |  |  |  |  |  |  |  |
|Our | 1 | 0.670 | 0.487 | 0.344 | 0.244 |  |  |  |  |
|Our | 3 | 0.677 | 0.497 | 0.364 | 0.270 |  |  |  |  |



The BLEU scores are 0.6706550835938225, 0.49190384936563414, 0.3608563774780518, 0.26851887441589206.The beam_size was 5.
The BLEU scores are 0.6623884069018237, 0.4838208582803625, 0.35399112652157766, 0.2637625748087203.The beam_size was 10.
The BLEU scores are 0.6574842336293149, 0.47854564809465855, 0.3494266073429842, 0.2604978054708325.The beam_size was 15.
The BLEU scores are 0.6553794319516962, 0.4764382019607749, 0.3473617206189104, 0.258826522877868.The beam_size was 20.


