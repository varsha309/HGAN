# ControlGAN 
Pytorch implementation for Controllable Text-to-Image Generation. The goal is to generate images from text, and also allow the user to manipulate synthetic images using natural language descriptions, in one framework. 

### Data

1. Download the preprocessed metadata for [bird](https://drive.google.com/file/d/1xdJEFqFxxw-2hkFGBOm5o13FwbQEjid6/view?usp=sharing) and save both into `data/`
2. Download [bird](https://drive.google.com/file/d/19RualH7lbYNY3AGDmYp8fJcpeXurJjH3/view?usp=sharing) dataset and extract the images to `data/birds/`

### Training
All code was developed and tested on ubuntu with Python 3.7 (Anaconda) and PyTorch 1.1.

#### [DAMSM]
- Pre-train DAMSM model for bird dataset:
```
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
```

#### ControlGAN model 
- Train ControlGAN model for bird dataset:
```
python main.py --cfg cfg/train_bird.yml --gpu 0
```


`*.yml` files include configuration for training and testing.


#### Pretrained DAMSM Model
- [DAMSM for bird](https://drive.google.com/drive/folders/1oMZm3TzCFyuXBtfm4DI-1Vqjsc8_CxqV?usp=sharing). Download and save it to `DAMSMencoders/`
#### Pretrained ControlGAN Model
- [ControlGAN for bird](https://drive.google.com/drive/folders/1fRo3Q4ALwoiFo2HQQAw1yG8yaSomiNB0?usp=sharing). Download and save it to `models/`

### Testing
- Test ControlGAN model for bird dataset:
```
python main.py --cfg cfg/eval_bird.yml --gpu 0
```

### Code Structure
- code/main.py: the entry point for training and testing.
- code/trainer.py: creates the main networks, harnesses and reports the progress of training.
- code/model.py: defines the architecture of ControlGAN.
- code/attention.py: defines the spatial and channel-wise attentions.
- code/VGGFeatureLoss.py: defines the architecture of the VGG-16.
- code/datasets.py: defines the class for loading images and captions.
- code/pretrain_DAMSM.py: creates the text and image encoders, harnesses and reports the progress of training. 
- code/miscc/losses.py: defines and computes the losses.
- code/miscc/config.py: creates the option list.
- code/miscc/utils.py: additional functions.


