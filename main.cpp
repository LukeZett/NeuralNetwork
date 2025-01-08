#include <random>
#include <cstdlib>
#include "Network/NetworkBuilder.h"
#include "Dataset/Dataset.h"
#include <iostream>

int main()
{
	std::filesystem::path dataPath = ".";
	Dataset train_labels = Dataset(dataPath / "fashion_mnist_train_labels.csv", 44);
	Dataset train_vectors = Dataset(dataPath / "fashion_mnist_train_vectors.csv", 44);

	srand(42);

	NN::NetworkBuilder builder = NN::NetworkBuilder({ 28,28,1 }, 10, 42);
	auto network = builder
		.Dense(128, NN::Trainable::ADAM)
		.Activation(NN::ActivationFunction::relu)
		.Dense(64, NN::Trainable::ADAM)
		.Activation(NN::ActivationFunction::relu)
		.Build(NN::Trainable::ADAM, NN::softmax);

	Dataset train_expected = train_labels.CreateLabelsVectors(10, 0);

	auto train_images_view = train_vectors.AsMatrix();
	auto train_expected_view = train_expected.AsMatrix();

	train_images_view /= 255;

	network.Learn(0.002, train_images_view, train_expected_view, 50, 20, 0.2);

	Dataset test_vectors = Dataset(dataPath / "fashion_mnist_test_vectors.csv", 44);
	Dataset test_labels = Dataset(dataPath / "fashion_mnist_test_labels.csv", 44);

	auto test_images_view = test_vectors.AsMatrix();
	test_images_view /= 255;

	uint32_t correct = 0;

	for (size_t i = 0; i < test_images_view.Height(); i++)
	{
		auto vecView = VecAl::VectorView(test_images_view.Width(), test_images_view[i]);
		if(NN::maxIndex(network.CalculateOutputs(vecView)) == test_labels[i][0])
		{
			correct++;
		}
	}

	std::cout << "Test set accuraccy: " << correct / (float)test_images_view.Height() << std::endl;
	return 0;
}