# merovingen-solver-v02

# [DESCRIPTION]
AI : Merovingen Solver Library

If you have any question about using this code or these algorythms for your own purposes, including adaptation to FPGA, subject to financing perspective technologes, or if you would like to financially support the project development, feel free to contact me with email or direct message.

Currently for the lack of ability to sign in for patenting the these algorythms, they are published for reviewing.

Special thanks to the person, who supported me all this time morally, but wished to remain without mentioning.

As an emerging breakthrough technology, artificial intelligence is releasing huge energy accumulated by the technological revolution and industrial transformation, profoundly changing the way of human production tools, lifestyle and thinking, and has a significant and far-reaching impact on economic development and progress

## [ENVIRONMENT]
```
* conda create --name tf2.15 python=3.11	# Create conda environment with tensorflow 2.15 version. Current version is not updapted for tensorflow 2.16 and keras versions.
* pip install tensorflow		# Or build from source: http://tensorflow.com
* pip install pillow			# Library to work with images
* pip install matplotlib 		# Library for plotting charts and diagrams
```

## [EXECUTE]
```
* conda activate tf2.15			# Activate prepared environment
* python ./plot_models.py		# Run the comparison test on the AI model training. Training resulting plots will be saved to ./traininfo
* python ./config.py			# Configuration file for the running test.
* python ./plot_activations.py	# Plots activation function comparison.
* python ./plot_spectre.py		# Plots spectre convertion and saves sample image to ./spectre.xxx.png
```

## [SPECTRE]

Algorythm for packing/unpacking encoding/decoding data/information with losses. The resulting test images will be saved to the ./spectre directory

python ./spectre/spectre.py


## [TRAINING RESULTS]

### Training summary:
![Training Summary](/train_results/Summary_2023-09-24-19-17.png?raw=true "Training Summary")

### Training details:
![Training Details](/train_results/Model_v02_IOL_V08_2023-09-03-07-57.png?raw=true "Training Details")

### Activation function comparison
![Activation Comparison](/train_results/activations_x_y_dx.png?raw=true "Activation Comparison")

### Spectre conversion
![Spectre Encoding](/spectre/birch.spectre.png?raw=true "Spectre Encoding")
