# show_tell
## Pytorch Implementation of Encoder-Decoder based Image Captioning. 
### The method used here is similar to _Show and Tell: Neural Image Caption (NIC) Generator model_

*Note:* This is a work in progress

In this work, I have implemented Ebcoder-Decoder based Image Captioning method. This method is similar to the method used in the paper: [Show and tell: A neural image caption generator](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html). This work here has been used as baseline in some of my research works where I have compared the performance of this method with other methods in the literature. Hence, to undertake a fair comparison, I have implemented this method with similar hyperparameter settings as other methods that have been studied or proposed by me. 
Thus, there are some differences in this implementation as compared to the method used in the paper. These are as follows:
1. In the paper (Show and tell: A neural image caption generator), authors use GoogLeNet CNN as encoder for image feature extraction but I have used VGG-16 as encoder.
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
|Paper | 20 | 0.666 | 0.461 | 0.329 | 0.246 |  |  |  |  |
|Our | 1 | 0.670 | 0.487 | 0.344 | 0.244 |  |  |  |  |
|Our | 3 | 0.677 | 0.497 | 0.364 | 0.270 |  |  |  |  |
|Our | 5 | 0.671 | 0.492 | 0.361 | 0.269 |  |  |  |  |
|Our | 10 | 0.662 | 0.484 | 0.354 | 0.264 |  |  |  |  |
|Our | 15 | 0.657 | 0.479 | 0.349 | 0.260 |  |  |  |  |
|Our | 20 | 0.655 | 0.476 | 0.347 | 0.259 |  |  |  |  |


Flickr8k RUN 5: using VGG-16 and 4096 sized image features from last FC layer
The model was trained for 13 epochs.
For the beam-size:1, the scores are:[('CIDEr', 0.44436728908787609), ('Bleu_4', 0.17304637090613423), ('Bleu_3', 0.2593040038962588), ('Bleu_2', 0.39303279313960027), ('Bleu_1', 0.5749009722722259), ('ROUGE_L', 0.43802327195046709), ('METEOR', 0.1928283224471941), ('SPICE', 0.12724625255034658)]
For the beam-size:3, the scores are:[('CIDEr', 0.4777015939873081), ('Bleu_4', 0.19848992923857056), ('Bleu_3', 0.2897326958019505), ('Bleu_2', 0.42357949851142807), ('Bleu_1', 0.6075569772091043), ('ROUGE_L', 0.44662664474909436), ('METEOR', 0.19155391555220577), ('SPICE', 0.13303113462890337)]
For the beam-size:5, the scores are:[('CIDEr', 0.47940450458617467), ('Bleu_4', 0.1972201319187718), ('Bleu_3', 0.28870704057167335), ('Bleu_2', 0.42172355374154463), ('Bleu_1', 0.6047637189661824), ('ROUGE_L', 0.44341522807548334), ('METEOR', 0.18639839652238166), ('SPICE', 0.13185361365682194)]
For the beam-size:10, the scores are:[('CIDEr', 0.46438612560673892), ('Bleu_4', 0.18715184658004963), ('Bleu_3', 0.27554082960373166), ('Bleu_2', 0.4087414567006016), ('Bleu_1', 0.5910507239787233), ('ROUGE_L', 0.43299182941967967), ('METEOR', 0.18199147570882598), ('SPICE', 0.12914096412226453)]
For the beam-size:15, the scores are:[('CIDEr', 0.45459571962082301), ('Bleu_4', 0.18326241625801876), ('Bleu_3', 0.2690186577145175), ('Bleu_2', 0.39851799214433126), ('Bleu_1', 0.5795117138582946), ('ROUGE_L', 0.4283237462152738), ('METEOR', 0.17755118005295803), ('SPICE', 0.12537116680371616)]
For the beam-size:20, the scores are:[('CIDEr', 0.45373528120505013), ('Bleu_4', 0.18174075103366252), ('Bleu_3', 0.2679371542707381), ('Bleu_2', 0.3971828497823847), ('Bleu_1', 0.5772579538529861), ('ROUGE_L', 0.42792423631911009), ('METEOR', 0.1768577144658102), ('SPICE', 0.12412007255608691)]

COCO RUN 3: Using VGG-16 4096 shaped features
The model was trained for 19 epochs.
For the beam-size:1, the scores are:[('CIDEr', 0.78736162897628892), ('Bleu_4', 0.2439521352046212), ('Bleu_3', 0.34456018369616775), ('Bleu_2', 0.4868389260279103), ('Bleu_1', 0.6704442054760499), ('ROUGE_L', 0.48997101687249667), ('METEOR', 0.22125809344669456), ('SPICE', 0.15320921346950706)]
For the beam-size:3, the scores are:[('CIDEr', 0.8215365774093728), ('Bleu_4', 0.27028380999699664), ('Bleu_3', 0.36398915301140394), ('Bleu_2', 0.49677080663603546), ('Bleu_1', 0.6768790328341467), ('ROUGE_L', 0.49808596550946121), ('METEOR', 0.22706190974027737), ('SPICE', 0.15785727740833672)]
For the beam-size:5, the scores are:[('CIDEr', 0.81552189484563253), ('Bleu_4', 0.26851887441588956), ('Bleu_3', 0.3608563774780486), ('Bleu_2', 0.49190384936562986), ('Bleu_1', 0.6706550835938169), ('ROUGE_L', 0.4949956855169218), ('METEOR', 0.22539232580402363), ('SPICE', 0.1570725486913844)]
For the beam-size:10, the scores are:[('CIDEr', 0.8021368252466764), ('Bleu_4', 0.26376257480871773), ('Bleu_3', 0.35399112652157433), ('Bleu_2', 0.48382085828035815), ('Bleu_1', 0.6623884069018178), ('ROUGE_L', 0.49037380934878638), ('METEOR', 0.22437412965830184), ('SPICE', 0.15394405694546698)]
For the beam-size:15, the scores are:[('CIDEr', 0.7915615358891942), ('Bleu_4', 0.2604978054708299), ('Bleu_3', 0.349426607342981), ('Bleu_2', 0.47854564809465416), ('Bleu_1', 0.6574842336293092), ('ROUGE_L', 0.48624022227109753), ('METEOR', 0.22239230359081671), ('SPICE', 0.1518799412402759)]
For the beam-size:20, the scores are:[('CIDEr', 0.78764923530989372), ('Bleu_4', 0.2588265228778655), ('Bleu_3', 0.3473617206189072), ('Bleu_2', 0.4764382019607707), ('Bleu_1', 0.6553794319516906), ('ROUGE_L', 0.48438604832296056), ('METEOR', 0.2215137586988965), ('SPICE', 0.15134202813526756)]

Flickr30k RUN 1: Using VGG-16 4096 shaped features
The model was trained for 18 epochs.
For the beam-size:1, the scores are:[('CIDEr', 0.32363713318297477), ('Bleu_4', 0.16556186979909188), ('Bleu_3', 0.24896640362699643), ('Bleu_2', 0.37583059865466945), ('Bleu_1', 0.5628349309381755), ('ROUGE_L', 0.41399811427817368), ('METEOR', 0.17594759546723385), ('SPICE', 0.11834172907256123)]
For the beam-size:3, the scores are:[('CIDEr', 0.36896157464809554), ('Bleu_4', 0.17915463659470393), ('Bleu_3', 0.26943348614942936), ('Bleu_2', 0.4040422547204555), ('Bleu_1', 0.6002813681526317), ('ROUGE_L', 0.42050501510906563), ('METEOR', 0.17282964381567228), ('SPICE', 0.11749876038592044)]
For the beam-size:5, the scores are:[('CIDEr', 0.38059434963615951), ('Bleu_4', 0.18593254722234137), ('Bleu_3', 0.2796573066559168), ('Bleu_2', 0.41962249807935537), ('Bleu_1', 0.6171623413525172), ('ROUGE_L', 0.41972348006167748), ('METEOR', 0.17217143660408796), ('SPICE', 0.11718108354005186)]
For the beam-size:10, the scores are:[('CIDEr', 0.38421632484615936), ('Bleu_4', 0.1936493505016226), ('Bleu_3', 0.28358502715416095), ('Bleu_2', 0.41911680242316507), ('Bleu_1', 0.6118451667272558), ('ROUGE_L', 0.41708400329816442), ('METEOR', 0.16779030265791536), ('SPICE', 0.11317668031096768)]
For the beam-size:15, the scores are:[('CIDEr', 0.37828176388389873), ('Bleu_4', 0.18386249929320597), ('Bleu_3', 0.2730612815483052), ('Bleu_2', 0.4093086841323574), ('Bleu_1', 0.6030711424969852), ('ROUGE_L', 0.41162236578219846), ('METEOR', 0.16362972315354724), ('SPICE', 0.10926752656260459)]
For the beam-size:20, the scores are:[('CIDEr', 0.37708869772704495), ('Bleu_4', 0.18480040429529), ('Bleu_3', 0.27287119765113294), ('Bleu_2', 0.4058877023904464), ('Bleu_1', 0.5959124160535607), ('ROUGE_L', 0.40975107383484155), ('METEOR', 0.16240473213291065), ('SPICE', 0.10846707894255288)]


