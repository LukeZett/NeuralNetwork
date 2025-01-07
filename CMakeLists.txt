cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork VERSION 0.0.0 
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(NeuralNetwork
	"Network/Network.cpp"
	"Network/Network.h"
	"NetworkLayers/BaseLayer/Layer.h"
	"NetworkLayers/BaseLayer/Trainable.h" 
	"NetworkLayers/BaseLayer/OutputLayer.h" 
	"NetworkLayers/DenseLayer/DenseLayer.h" 
	"NetworkLayers/DenseLayer/DenseLayer.cpp" 
	"NetworkLayers/DenseLayer/DenseLayerTrain.h" 
	"NetworkLayers/DenseLayer/DenseLayerTrain.cpp" 
	"Components/Optimizers/SGD.h" 
	"Components/Optimizers/ADAM.h" 
	"Components/Optimizers/Optimizer.h"
	"Components/CostFunctions/CostFunction.h" 
	"Components/CostFunctions/MSE.h" 
	"NetworkLayers/ActivationLayer/ActivationLayer.h" 
	"NetworkLayers/ActivationLayer/ActivationLayerTrain.h" 
	"Components/ActivationFunctions/Sigmoid.h" 
	"Network/NetworkBuilder.h" 
	"Network/NetworkBuilder.cpp"
	"Components/ActivationFunctions/ReLU.h"
	"Components/ActivationFunctions/LeakyReLU.h" "NetworkLayers/SoftMaxLayer/SoftMaxLayer.h" "NetworkLayers/SoftMaxLayer/SoftMaxLayerTrain.h" "Components/CostFunctions/CrossEntropy.h" "NetworkLayers/Conv2DLayer/Conv2DLayer.h" "NetworkLayers/Conv2DLayer/Conv2DLayerTrain.h" "NetworkLayers/Conv2DLayer/Conv2DLayer.cpp" "NetworkLayers/Conv2DLayer/Conv2DLayerTrain.cpp" "NetworkLayers/Maxpool2DLayer/MaxPool2DLayer.h" "NetworkLayers/Maxpool2DLayer/MaxPool2DLayer.cpp")

target_link_libraries(NeuralNetwork PUBLIC VecAl)

target_include_directories(NeuralNetwork PUBLIC "NetworkLayers")
target_include_directories(NeuralNetwork PUBLIC "Components")
target_include_directories(NeuralNetwork PUBLIC "Network")
target_include_directories(NeuralNetwork PUBLIC ".")


if (MSVC)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		
	else()
		target_compile_options(NeuralNetwork PRIVATE /O2 /Ob3 /fp:fast)
	endif()

endif()