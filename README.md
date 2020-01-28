# Toy dataset based on moving MNIST to evaluate models for Multi-Object Tracking and Segmentation

**This is work in progress!!**

This is an extension of the moving MNIST dataset by [1] based on the implementation by [praateekmahajan](https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe) (thank you!).

It is tailored towards the Multi-Object Tracking and Segmentation (MOTS) task introduced by [2].

This includes the following modifications:

- Numbers have a fixed order with regard to foreground to background
- Numbers change their size while moving
- Numbers change their appearance while moving
- (todo) Numbers can disappear in a sequence and new numbers can enter

The generated data can be exported in COCO format [3] or MOTS format [2].


# Installation

```
python3 -m venv path/to/venvs/moving-mnist-mots         # Create a virtual environment
source path/to/venvs/moving-mnist-mots/bin/activate     # Activate the virtual environment
pip install -r requirements.txt
pip install pycocotools
```

# Configuration

The configuration is controlled by `sacred`. For options see [config.yaml](config.yaml).

# Generate data

```
python3 moving_mnist.py
```

# References

[1] Srivastava, N., Mansimov, E., & Salakhudinov, R. (2015, June). Unsupervised learning of video 
    representations using lstms. In International conference on machine learning (pp. 843-852).

[2] Voigtlaender, P., Krause, M., Osep, A., Luiten, J., Sekar, B. B. G., Geiger, A., & Leibe, B. 
    (2019). MOTS: Multi-object tracking and segmentation. In Proceedings of the IEEE Conference on 
    Computer Vision and Pattern Recognition (pp. 7942-7951)

[3] Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. 
    (2014, September). Microsoft coco: Common objects in context. In European conference on 
    computer vision (pp. 740-755). Springer, Cham.