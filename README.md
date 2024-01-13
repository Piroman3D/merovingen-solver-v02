l# merovingen-solver-v02

# [DESCRIPTION]
AI : Merovingen Solver Library

Для искушённых научной фантастикой читателей документации, данный проект может позиционироваться как дрова на гравицапу, из не мало известного фильма "Кин Дза-Дза"

цитата из фильма:
  ― Вы взрослый человек, Гедеван Александрович. Вы проучились один семестр и исчезли на годы. Объявились с каким-то камешком, с каким-то обломком кавказской керамики и (звенит) колокольчиком от донки, а претендуете на… Ну, и к тому же ― если вы способны музицировать, тогда почему вы не принимали участия в нашей курсовой самодеятельности? Вы извините меня, скрипач, но это элементарное ку!


This project is related to neural network functionality and implements range of custom layers that together helps to achieve higher efficiecy during neural network training process. The core neural network consists of the follwing layers:
1. IOLActivation | ./layers/iolactivation_v08.py
	Custom activation function as a replacement for siren, snake, and relu activations.
2. MemBlock 	 | ./layers/memblock.py 
	Functional layer implementing memory functionality for the neural network. The layer affects the neural network both at input and output of data, that enchances training capabilities of the network. On each step the neural network is self adjusting it's weights located in  memory allocated area. As well as after training step backpropagation process helps to adjust its weight too in combination with model self adjustment.
3. ModulatedConv2D | ./layers/modconv.py 
	Modulated convolution layer based on the x input and style input, the layer is based on Nvidia solution for GANs, but features different implementation to achieve better compatibility with other layers and focuses on 1D modulated convolution, for 1D signal processing.
4. PhyBlock | ./layers/phyblock.py
	Physical constant memory storage that is passed through densely connected layer, to achieve required ratios of the physical constants that can be involved in the data processing.
5. Snake 	| ./layers/snake.py	
	Activation function by Tensorflow authors for comparison of the approaches and results.
6. SpectralDense | ./spectredense.py
	Spectral Dense layer, based on the densely connected layer and ./spectre/spectre.py algorythm to reduce the weight count required for traing.

If you have any question about using this code or these algorythms for your own purposes, including adaptation to FPGA, subject to financing perspective technologes, or if you would like to financially support the project development, feel free to contact me with email or direct message.

Currently for the lack of ability to sign in for patenting these algorythms, they are published for reviewing.

Special thanks to the person, who supported me all this time morally, but wished to remain without mentioning.

As an emerging breakthrough technology, artificial intelligence is releasing huge energy accumulated by the technological revolution and industrial transformation, profoundly changing the way of human production tools, lifestyle and thinking, and has a significant and far-reaching impact on economic development and progress

## [ENVIRONMENT]
```
# Create conda environment with tensorflow 2.15 version.
# Current version is not updapted for tensorflow 2.16 and keras versions.

* conda create --name tf2.15 python=3.11
* pip install tensorflow==2.15		# Or build from source: http://tensorflow.com
* pip install pillow			        # Library to work with images
* pip install matplotlib 		      # Library for plotting charts and diagrams
```

## [EXECUTE]
```
* conda activate tf2.15			# Activate prepared environment
* python ./plot_models.py		# Run the comparison test on the AI model training. Training resulting plots will be saved to ./traininfo
* python ./plot_activations.py	# Plots activation function comparison.
* python ./plot_spectre.py		# Plots spectre convertion and saves sample image to ./spectre.xxx.png
* python ./config.py			# Configuration file for the running test.
```

## [SPECTRE]

Algorythm for packing/unpacking encoding/decoding data/information with losses. The resulting test images will be saved to the ./spectre directory

```
python ./spectre/spectre.py
```

## [TRAINING RESULTS]

### Training summary:
Green samples shows predicted function. Blue samples shows the dataset the neural network was trained.
Current dataset contains prediction samples of 1D sinusoidal wave. And was trained to predict sinus wave samples.

Horizontal axis is training time in seconds. The grey line shows the fastest fit of the dataset.
![Training Summary](/train_results/Summary_2023-09-24-19-17.png?raw=true "Training Summary")

### Training details:
In many trained cases, the algorythm showed good ability to extrapolate dataset. Initial dataset contains training information only in range from -1.0 to 1.0
![Training Details](/train_results/Model_v02_IOL_V08_2023-09-03-07-57.png?raw=true "Training Details")

### Activation function comparison
Activation function comparison. Blue line is the activation function, and violet line is it's first derivative
![Activation Comparison](/train_results/activations_x_y_dx.png?raw=true "Activation Comparison")

### Spectre conversion
Sample image of the information compression by the spectre algroythm. This can be considered as the method for quatization/dequantization. Currently the algorythm is not used during the training process, but should significantly reduce amount redundand information during neural network training.
![Spectre Encoding](/spectre/birch.spectre.png?raw=true "Spectre Encoding")

## [NOTICE]
The Merovingen Solver source code or its parts and derivatives are prohibited for usage in harmful actions or actions that can lead directly or indirectly to harm or damage yourselves or others, including harmful or damaging actions in financial, civil relations and other fields.
