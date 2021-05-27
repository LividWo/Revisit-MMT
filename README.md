# Good for Misconceived Reasons: An Empirical Revisiting on the Need for Visual Context in Multimodal Machine Translation

<!-- []() -->

This repo contains code needed to replicate our findings in the ACL’2021 paper:


## Setup conda environment (recommanded)
- conda create --name revisit python=3.7.6
- conda activate revisit
- pip install transformers==3.0.2
- conda install pytorch\==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
- cd Revisit-MMT/
- pip install --editable ./  


## Example Usage
```bash
#We have included pre-processed raw data from Multi30K En-De in the repo, with the following format (take train set as an example):

train.en # source sentence
train.de # target sentences
train.vision.en # image id associated with each source sentence
# We re-index image id from 0 to #number of images for convinence
train.bert.en # the source sentence tokenized by bert tokenizer, used for retrieval
```




[Transformer basleine on Multi30K En-De](https://github.com/LividWo/Revisit-MMT/blob/master/README-baseline.md)

[Gated Fusion on Multi30K En-De](https://github.com/LividWo/Revisit-MMT/blob/master/README-gated.md)
PS: you need download pre-extracted visual features to train a Gated Fusion model.

[RMMT on Multi30K En-De](https://github.com/LividWo/Revisit-MMT/blob/master/README-RMMT.md)
PS: you need download pre-extracted visual features and a pre-trained image retriever to tran RMMT.


## Resources 
File Name | Description |  Download
---|---|---
`resnet50-avgpool.npy` | pre-extracted image features, each image is represented as a 2048-dimensional vector. | 
`retriever.bin` | pre-trained text->image retriever | 
`checkpoints` | pre-trained Transformer/Gated Fusion/RMMT on Multi30K En-De for quick reproduction | 
