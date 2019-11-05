# Computer Vision Tasks
1. Semantic Segmentation: no objects, just pixels

    Solution: fully convolutinal, design a network as a bunch of convolutional layers to make predictions for pixels all at once. 
    -   with downsampling and upsampling in side the network
    -   upsampling: nearest neighbor, bed of nails, max unpooling, transpose convolution
    -   why upsampling: help us find details better, preserve some information that was lost

2. Classification + Localization: single object

    Solution: fully connected will generate two out comes, Class Score AND Box Coordinates
    -   Softmax Loss for correct label
    -   L2 loss for correct box

3. Object Detection: multiple object: different from classification + localization problem, because there might be a varying number of outputs for every input image

    Solution: firstly, find blobby image regions that're likely to contain objects, relatively fast to run, for example, selective search gives 2000 region proposals in a few seconsds on CPU, to get some set of proposal regions where objects are likely located. secondly, apply a convolutional network for classification to each of these proposal regions and this will end up being much more compytationally tractable than trying to do all possible locaation and scales

    Method: R-CNN, Fast R-CNN, Faster R-CNN, YOLO

    a. R-CNN: input image, extract region. warped region, compute cnn feature, classify regions, improve bounding boxes
    
    b. Fast R-CNN:

    -   https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e
    -   https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4
    
4. Instance Segmentation: multiple object