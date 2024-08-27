# Robust Encrypted Inference in Deep Learning: A Pathway to Secure Misinformation Detection
To combat the rapid spread of misinformation on social networks, automated misinformation detection systems based on deep neural networks (DNNs) have been developed. However, these tools are often proprietary and lack transparency, which limits their usefulness. Furthermore, privacy concerns limit data sharing by data owners as well as by data-driven misinformation-detection services. Although data encryption techniques can help address privacy concerns in DNN inference, there is a challenge to the seamless integration of these techniques due to the encryption errors induced by cascaded encrypted operations, as well as a mismatch between the tools used for DNNs and cryptography. In this paper, we make two-fold contributions. Firstly, we study the noise bounds of homomorphic encryption (HE) operations as error propagation in DNN layers and derive two properties that, if satisfied by the layer, will considerably reduce the output error. We identify that L2 regularization and sigmoid activation satisfy these properties and validate our hypothesis, for instance, replacing ReLU with sigmoid reduced the output error by 106× (best case) to 10× (worst case). Secondly, we extend the Python encryption library TenSeal by enabling the automatic conversion of a TensorFlow DNN into an encryption-compatible DNN with a few lines of code. These contributions are significant as encryption-friendly DL architectures are sorely needed to close the gap between DL-in-research and DL-in-practice.


## How to use the code?
```
python main.py
```
(More instructions on how to setup will be uploaded soon.)



## Cite as:
```
@article{ali2024robust,
  title={Robust Encrypted Inference in Deep Learning: A Pathway to Secure Misinformation Detection},
  author={Ali, Hassan and Javed, Rana Tallal and Qayyum, Adnan and AlGhadhban, Amer and Alazmi, Meshari and Alzamil, Ahmad and Al-utaibi, Khaled and Qadir, Junaid},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2024},
  publisher={IEEE}
}
```