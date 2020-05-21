# DeepEEG
Deep feature learning using a specially designed __CNN__ for processing raw EEG data in order to predict MDD patient's response to SSRI treatment.

In this project, we have proposed two new Convolution blocks which aim to reduce parameters of conventional Temporal Convolution blocks which makes it plausible to train deep CNN models to classify multi-variate time-series data constrained with limited amount of training data, which is the case in classifying resting state EEG signals. These blocks named as **WFB** (Windowed Filter-Bank) and **DFB** (Dilated Filter-Bank) are designed to force the model learn spatio-temporal features that capture the frequency components of various frequency bands. DFB mimics the wavelet transform but as a learnable spatio-temporal filter-bank, in the other side the WFB mimics the FFT in extracting various frequency components from a window, but it is a learnable transforma and the earned features are spatio-temporal rather than temporal. We show that the DFB blocks are superior in terms of improving accuracy, f1-score and decreasing the number of parameters.

Comparing the architecture design choices proves the superiority of wavelet-like Conv1D blocks (results on _Repponder_ vs. _Non-Responder_ task):


| Model        | # **Trainable Params**           | **F1-Score** (std)  |
| -------------|:-------------:| -----:|
| _BaselineDeepEEG_      | 70,668 | 0.25 (0.32) |
| _DilatedDeepEEG_  (DFB)  | 8,804 | **0.66** (0.17) |
| _LightDilatedDeepEEG_  (DFB)   | **4,408** | 0.63 (0.16) |
| _WindowedDeepEEG_ (WFB)      | 41,228 | 0.63 (0.17) |




Here's a simple schema which shows the overall model's architecture:

![alt text](https://github.com/iamsoroush/DeepEEG/blob/master/deep_eeg_arch.jpg "DeepEEG Architecture")



And here's an intuitive presentation of how **DFB** and **WFB** work:

![alt text](https://github.com/iamsoroush/DeepEEG/blob/master/st-dfb-wfb.jpg "ST-DFB and ST-WFB")


# How to use
To reproduce the results, do the following:
1. Create a copy from [*prepare_data*](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/prepare_data.ipynb) in your google drive.
2. Run all cells within *Copy of prepare_data.ipynb* one-by-one.
2. Create a copy from a *cv_....ipynb* file in your google drive.
4. Run cells one-by-one, and you got the results!



# Results

We have compared our model on _Repponder_ vs. _Non-Responder_ task with some other models that we have found which are end-to-end learnable:

| Model        | # **Trainable Params**           | **F1-Score** (std)  |
| -------------|:-------------:| -----:|
| [_E-ST-CNN_](https://ieeexplore.ieee.org/document/8607897)      | 153,093 | 0.4 (0.09) |
| [_Conv2DModel_](https://onlinelibrary.wiley.com/doi/10.1002/spe.2668)      | 3,550,069 | 0.03 (0.11) |
| [_EEGNet_](https://arxiv.org/abs/1611.08024)      | **2,305** | 0.6 (0.2) |
| _DeepEEG_ (ours)(DFB + TemporalAttention)   | 8,804 | **0.67** (0.15) |

