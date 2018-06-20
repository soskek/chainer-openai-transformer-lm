# Chainer implementation of OpenAI's Finetuned Transformer Language Model

This is a **Chainer** implementation of the [TensorFlow code](https://github.com/openai/finetune-transformer-lm) provided with OpenAI's paper ["Improving Language Understanding by Generative Pre-Training"](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.

This implementation comprises **a script to load in the Chainer model the weights pre-trained by the authors** with the TensorFlow implementation.
This is made from [pytorch implementation](https://github.com/huggingface/pytorch-openai-transformer-lm) by line-level replacements as possible.
If you are interested, see [the diff](https://github.com/soskek/chainer-openai-transformer-lm/commit/b2b971e460e66d8318c2ff0c1b48621856509673).
This does not always contain implementations which are conventionally natural for chainer, but you can enjoy alignments with pytorch (and tensorflow).

This implementation achieved better or almost same accuracies than the original one, on the ROCStories test set. (Med: 85.81 vs 85.8, Best: 87.33 vs 86.5 in 10 runs.)

![Transformer Language Model](assets/ftlm.png)

The model classes and loading script are located in [model_py.py](model_py.py).

The names of the modules in the Chainer model follow the names of the Variable in the TensorFlow implementation. This implementation tries to follow the original code as closely as possible to minimize the discrepancies.

This implementation thus also comprises a modified Adam optimization algorithm as used in OpenAI's paper with:
- fixed weights decay following the work of [Loshchilov et al.](https://arxiv.org/abs/1711.05101), and
- scheduled learning rate as [commonly used for Transformers](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer).

## Requirements
To use the model it-self by importing [model_py.py](model_py.py), you just need:
- Chainer
- [cupy](https://github.com/cupy/cupy) (for gpu run)

To run the classifier training script in [train.py](train.py) you will need in addition:
- tqdm
- sklearn
- spacy
- ftfy
- pandas

You can download the weights of the OpenAI pre-trained version by cloning [Alec Radford's repo](https://github.com/openai/finetune-transformer-lm) and placing the `model` folder containing the pre-trained weights in the present repo.
```bash
sh download_model_params.sh
```


## Using the pre-trained model as a Transformer Language Model
The model can be used as a transformer language model with OpenAI's pre-trained weights as follow:
```python
from model_py import Model, load_openai_pretrained_model, DEFAULT_CONFIG

args = DEFAULT_CONFIG
model = Model(args)
load_openai_pretrained_model(model)
```

This model generates Transformer's hidden states. You can use the `LMHead` class in [model.py](model.py) to add a decoder tied with the weights of the encoder and get a full language model. You can also use the `ClfHead` class in [model.py](model.py) to add a classifier on top of the transformer and get a classifier as described in OpenAI's publication. (see an example of both in the `__main__` function of [train.py](train.py))

To use the positional encoder of the transformer, you should encode your dataset using the `encode_dataset()` function of [utils.py](utils.py). Please refer to the beginning of the `__main__` function in [train.py](train.py) to see how to properly define the vocabulary and encode your dataset.

## Fine-tuning the pre-trained model on a classification task
This model can also be integrated in a classifier as detailed in [OpenAI's paper](https://blog.openai.com/language-unsupervised/). An example of fine-tuning on the ROCStories Cloze task is included with the training code in [train.py](train.py)

The ROCStories dataset can be downloaded from the associated [website](http://cs.rochester.edu/nlp/rocstories/).

As with the [TensorFlow code](https://github.com/openai/finetune-transformer-lm), this code implements the ROCStories Cloze Test result reported in the paper which can be reproduced by running:

```bash
python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir [path to data here]
```

#### First experiments on the ROCStories test set

Test accuracies from 10 runs were [54.04, 84.02, 85.2, 85.62, 85.73, 85.89, 86.26, 86.53, 86.91, 87.33]. The median is 85.81, which is better than the score [original tensorflow code](https://github.com/openai/finetune-transformer-lm) reported (85.8). The best score is 87.33, which is also better than the score in the paper (86.5).

Note that these exactly same values cannot be reproduced because GPU calculations are often not deterministic.

For throughput during training, this chainer version achieved around 3.8 it/s while the pytorch code got around 1.1 it/s, in my environment using Tesla P100.
