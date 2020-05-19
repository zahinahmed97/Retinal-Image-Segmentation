Steps to run training:

1. Ensure the "train" and "test" folders of the DRIVE dataset are in this folder
2. Run the prep_datasets.py folder to prepare the hdf5 files
3. Run the train_UNET.py to train U-Net model or train_BCDU.py to train BCDU-Net model

Steps to run testing:

1. Run the predict_UNET.py to test images using U-Net model
2. Run the predict_BCDU.py to test images using BCDU-Net model

Preprocessing code and details are in the pre_processing file. The steps followed are: RGB to greyscale conversion, normalising, Contrast limited adaptive histogram equalisation, gamma adjustment and intensity scaling.

Other changeable parameters (batch size, number of patches etc) are in the training files 

Code used for U-Net model and preprocessing: https://github.com/orobix/retina-unet
Code used for BCDU-Net model: https://github.com/rezazad68/BCDU-Net

References:
[1] O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", Lecture Notes in Computer Science, pp. 234-241, 2015. Available: 10.1007/978-3-319-24574-4_28 

[2] R. Azad, M. Aghbolaghi, M. Fathy and S. Escalera, "Bi-Directional ConvLSTM U-Net with Densley Connected Convolutions", 2019
