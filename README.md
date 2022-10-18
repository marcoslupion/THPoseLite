# THPoseLite
This repository incorporates THPoseLite, a neural network that obtains the pose from a person in thermal images. 

## Usage

Run the script *THPoseLite.py* with the parameter *--name* indicating the neural network that will be created. Options are:

- ResNet50
- UNET
- MobileNetV2

In the folder [*TrainedModels/*](https://drive.google.com/drive/folders/1BSizNKKjcjbCyk_uvri4VKW0neZ7fsxE?usp=sharing) from Google Drive, pre-trained models with a dataset recorded in the University of Almería are incorporated. For each of the model, there are three configurations:
- Tensorflow model. Weights are stored in 64-bit precision format. 
- TFLite model. Model that can be incorporated in smartphones and embedded devices. 
- Quantized model. Weights are stored in 8-bit precision format. 


## License
This software is provided 'as is', with the best of intentions but with any kind of warranty and responsibility of stability and correctness. It can be freely used, distributed and modified by anyone interested at it but a reference to this source is required.

## Contact

For any question or suggestion feel free to contact:

- Marcos Lupión: [marcoslupion@ual.es](marcoslupion@ual.es)