# Understanding Deep Face Representation via Attribute Recovery

Deep neural networks have proven to be highly effective in the face recognition task, as they can map raw samples into a discriminative high-dimensional facial representation space. However, understanding this complex space proves to be challenging for human observers. In this paper, we propose a novel approach that interprets deep face recognition models via facial attributes. To achieve this, we introduce a two-stage framework that recovers attributes from the deep face representations. This framework allows us to quantitatively measure the significance of facial attributes in relation to the recognition model. Moreover, this framework enables us to generate image-specific explanations through the use of counterfactual samples. These explanations are not only understandable but also quantitative. Through the proposed approach, we are able to acquire a deeper understanding of how the recognition model conceptualizes the notion of ``identity" and understand the reasons behind errors the model may make. By utilizing attributes as an interpretable medium, the proposed method marks a paradigm shift in our comprehension of deep face recognition models. It allows a complex model, obtained through gradient backpropagation, to effectively ``communicate" with humans.

![arch](method.png)

## The proposed framework

![arch](framework_infe.png)

# Usage Instructions

## Requirments

python == 3.7

pytorch == 1.6.0

torchvision == 0.7.0

bcolz == 1.2.1

tqdm



## Facial image Decoding

`lfw_recon.py ` to decode facial image.

Pretrained model can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1pcF7fq4JtoQQzraq3TR5-w?pwd=rtwp), code: rtwp
or [Google Drive](https://drive.google.com/drive/folders/1mDMeZgqTiaoOvDN6DhFQ7t45IOVAV5jX?usp=sharing)

# Citation
If you find our work useful in your research, please consider to cite:

    @ARTICLE{10587012,
    author={Ren, Min and Zhu, Yuhao and Wang, Yunlong and Huang, Yongzhen and Sun, Zhenan},
    journal={IEEE Transactions on Information Forensics and Security}, 
    title={Understanding Deep Face Representation via Attribute Recovery}, 
    year={2024},
    volume={},
    number={},
    pages={1-1},
    keywords={Face recognition;Facial features;Deep learning;Analytical models;Visualization;Nose;Uncertainty;Interpretability;face recognition;facial attribute;counterfactual sample},
    doi={10.1109/TIFS.2024.3424291}}
