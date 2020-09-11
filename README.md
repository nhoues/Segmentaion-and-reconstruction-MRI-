# Segmentaion-and-reconstruction-MRI

**This work was created for our engineering internship at BASIRA lab and L3S ENIT. It contains a reconstruction of 3T HR-like MR images of the hippocampus from 3T LR MR images and segmentation of the hippocampal subfields form 3T MR images. In this work we used Deep Learning models independently and then jointly to enhance both reconstruction and segmentation.** 

* UNet for Reconstruction : implementation of UNet to reconstitue a 3T HR-like image.
* UNet for Segmentation: implementation of UNet to segment hippocampal subfields.
* GAN based on UNet for reconstruction: implementation of GAN based on UNet to reconstitute a 3T HR-like image.
* GAN based on UNet for segmentation: implementation of GAN based on UNet to segment hippocampal subfields.
* MUNet : implementation of a model based on UNet which inputs a 3T LR image and outputs the 3T HR-like and the segmented hippocampal subfields.
* C-UNet : implementation of 4 cascaded blocs of UNet which train alternatively on reconstruction and segmentation.
* C-UNet V2 : implementation of 5 cascaded blocs of UNet which train alternatively on reconstruction and segmentation.
* PC-UNet : implementation of 2 paralleled blocs of 2 cascaded blocs of UNet which train jointly reconstruction and segmentation.
* PC-UNet V2 : implementation of 2 paralleled blocs of 3 cascaded blocs of UNet which train jointly reconstruction and segmentation.
* GAN based on C-UNet : implementation of GAN based on 5 cascaded blocs of UNet which train alternatively on reconstruction and segmentation.
* GAN based on PC-UNet : implementation of GAN based on 2 paralleled blocs of 3 cascaded blocs of UNet which train jointly reconstruction and segmentation.
* summary-Segmentation : barplots showing dice score for each hippocampal segment for all models.
* summary-reconstruction : barlpots showing SSIM and PSNR scores of reconstruction for all models.

