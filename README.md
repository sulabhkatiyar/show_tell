# show_tell
## Pytorch Implementation of Encoder-Decoder based Image Captioning. 
### The method used here is similar to _Show and Tell: Neural Image Caption (NIC) Generator model_

*Note:* This is a work in progress. More CNN encoders can be used and results will be updated soon. 

#### Description:

In this work, I have implemented Encoder-Decoder based Image Captioning method. This method is similar to the method used in the paper: [Show and tell: A neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html). This work here has been used as baseline in some of my research works where I have compared the performance of this method with other methods in the literature. Hence, to undertake a fair comparison, I have implemented this method with similar hyperparameter settings as other methods that have been studied or proposed by me. 
Thus, there are some differences in this implementation as compared to the method used in the paper. These are as follows:
1. In the paper (Show and tell: A neural image caption generator), authors use GoogLeNet CNN as encoder for image feature extraction but I have used VGG-16 as encoder.
1. I have not used model ensembles in this implementation. The authors determine that using model ensembles helps them enhance the performance with a few points on most BLEU metrics.
1. I have not used batch normalization for inputs. I have observed that, using batch normalization for VGG-16 CNN provides inferior results.
1. Beam width of 3 has been used for inference. I have observed that, as compared to beam width of 20 which has been used in this paper, beam width of 3 provides better results. While using our method, beam width of 20 also provides results comparable to the results in the paper. Here, I have quoted the results with all beam sizes, for the sake of completeness.

All the results are not available from the paper. Hence, some results have been quoted from the paper: [Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention](http://proceedings.mlr.press/v37/xuc15) where the authors obtain the results after personal corespondence with the authors of _Show and tell: A neural image caption generator_.

As we can observe, the results for Flickr8k and Flickr30k datasets are inferior, for most of the metrics, as compared to the results in the paper: _Show and tell: A neural image caption generator_. This seems to be due to use of model ensembles in the paper. For MSCOCO, our method provides slightly better results for most metrics. This may be partly due to the fact that authors cite the results obtained from evaluation with a beam_size of 20 and also due to the fact that with a larger training data available (MSCOCO has over 100k images as compared to 8k for Flickr8k and 30k for Flickr30k dataset) the use of model ensembles does not provide significant boost in performance.

One important observation is that our method provides lower values on all evaluation metrics for Flickr30k dataset as compared to Flickr8k dataset even though the former contains more training samples. We have observed the same for other methods proposed in the literature as well (such as the Visual Attention mechanism proposed in _Show, Attend and Tell: Neural Image Caption Generation with Visual Attention_). This may be due to structure of dataset (test set images or captions may be significantly differrent than training set) although I have not seen any research on the reason behind it.

**For Flickr8k dataset:**

|Result |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
|Paper | 20 | 0.63 | 0.41 | 0.27 |  | __ | __ | __ | __ |
|Our | 1 | 0.575 | 0.393 | 0.259 | 0.173 | 0.193 | 0.444 | 0.127 | 0.438 |
|Our | 3 | 0.608 | 0.424 | 0.290 | 0.198 | 0.191 | 0.478 | 0.133 | 0.447 |
|Our | 5 | 0.605 | 0.421 | 0.289 | 0.197 | 0.183 | 0.479 | 0.132 | 0.443 |
|Our | 10 | 0.591 | 0.409 | 0.275 | 0.187 | 0.182 | 0.464 | 0.129 | 0.433 |
|Our | 15 | 0.580 | 0.399 | 0.269 | 0.183 | 0.178 | 0.454 | 0.125 | 0.428 |
|Our | 20 | 0.577 | 0.397 | 0.268 | 0.182 | 0.177 | 0.454 | 0.124 | 0.428 |

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
|Paper | 20 | 0.666 | 0.461 | 0.329 | 0.246 |  |  |  |  |
|Our | 1 | 0.670 | 0.487 | 0.344 | 0.244 | 0.221 | 0.787 | 0.153 | 0.490 |
|Our | 3 | 0.677 | 0.497 | 0.364 | 0.270 | 0.227 | 0.821 | 0.158 | 0.498 |
|Our | 5 | 0.671 | 0.492 | 0.361 | 0.269 | 0.225 | 0.816 | 0.157 | 0.495 |
|Our | 10 | 0.662 | 0.484 | 0.354 | 0.264 | 0.224 | 0.802 | 0.154 | 0.490 |
|Our | 15 | 0.657 | 0.479 | 0.349 | 0.260 | 0.222 | 0.791 | 0.152 | 0.486 |
|Our | 20 | 0.655 | 0.476 | 0.347 | 0.259 | 0.221 | 0.788 | 0.151 | 0.484 |


#### Prerequisites:
1. This code has been tested on python 3.6.9 but should word on all python versions > 3.6.
2. Pytorch v1.5.0
3. CUDA v10.1
4. Torchvision v0.6.0
5. Numpy v.1.15.0
6. pretrainedmodels v0.7.4 (Install from [source](https://github.com/Cadene/pretrained-models.pytorch.git)). (I think all versions will work but I have listed here for the sake of completeness.)


#### Execution:
1. First set the path to Flickr8k/Flickr30k/MSCOCO data folders in create_input_files_dataname.py file ('dataname' replaced by f8k/f30k/coco).
2. Create processed dataset by running: 
> python create_input_files_dataname.py

3. To train the model:
> python train_dataname.py

4. To evaluate: 
> python eval_dataname.py beamsize 

(eg.: python train_f8k.py 20)
