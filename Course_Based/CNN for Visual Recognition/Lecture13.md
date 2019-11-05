# Lecture 13 | Generative Models

Generative models are a subset of unsupervised learning wherein given some training data we generate new samples/data from the same distribution, like modeling the distribution of natural images .

Supervised Learning: classification, regresion, object detection, semantic segmantation, image captioning. Unsupervised Learning: clustering, dimensionality reduction, feature learning, density estimation.

Generateive Models
- Auto-Regressive (PixelRNN and PixelCNN)
- Variational Autoencoders(AVE)
- Generative Adversarial Networks(GAN)

Why Generative Models?
- Realistic samples for artwork, super-resolution, colorization
- Generative models of time series data can be used for simulation and planning(reinforcement learning applications!)
- Training generative models can also enable inference of latent representations that can be useful as general features

Application of Generative Models: image deblurring image compression, text to image conversion. 

The difference between GANs and Auto-regressive models is that GANs learn implicit data distribution whereas the latter learns an explicit distribution governed by a prior imposed by model structure. 
Advantage of Auto regressive:
- provide a way to calculate likelihood
- training is more stable than GANs
- works for both discrete and continuous data
GANs are known to produce higher quality images and are faster to train


An effective approach to model such a network is to use probabilistic density models (like Gaussian or Normal distribution) to quantify the pixels of an image as a product of conditional distributions. 
This approach turns the modeling problem into a sequence problem wherein the next pixel value is determined by all the previously generated pixel values.

PixelRNN 

    The network scans the image one row one pixel at a time in each row. Subsequently it predicts conditional distributions over the possible pixel values. The distribution of image pixels is written as product of conditional distributions and these values are shared across all the pixels of the image.
    The objective here is to assign a probability p(x) to every pixel of the (n x n) image. Since we can now know the conditional probability of our pixel value, to get the appropriate pixel value we use a 256-way softmax layer. Output of this layer can take any value ranging from 0–255. Here, negative log likelihood (NLL) is used as the loss and evaluation metric as the network predicts(classifies) the values of pixel from values 0–255.

- Other types of architectures which can be used: Row LSTM, Diagonal BiLSTM, a fully convolutional network and a Multi Scale network.

PixelCNN
   
    PixelCNN uses standard convolutional layers to capture a bounded receptive field and compute features for all pixel positions at once. It uses multiple convolutional layers that preserve the spatial resolution. However, pooling layers are not used. Masks are adopted in the convolutions to restrict the model from violating the conditional dependence.

    Problem: 
   
    PixelCNN reduces the computational cost required in Row LSTM and diagonal BLSTM but suffers from the problem of blind spot. Blind spot problem is basically not covering all the previous pixels in the context/history used to compute the hidden state of a pixel. 


1. https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173
2. https://towardsdatascience.com/summary-of-pixelrnn-by-google-deepmind-7-min-read-938d9871d6d9
3. http://sergeiturukin.com/2017/02/22/pixelcnn.html