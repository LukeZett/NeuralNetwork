#include <random>
#include <cstdlib>
#include "Network/NetworkBuilder.h"
#include "Dataset/Dataset.h"
#include <iostream>
#include <fstream>

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

	auto test_images_view = test_vectors.AsMatrix();
	test_images_view /= 255;

	std::ofstream trainPred("../../../../train_predictions.csv");
	std::ofstream testPred("../../../../test_predictions.csv");

	for (size_t i = 0; i < test_images_view.Height(); i++)
	{
		auto vecView = VecAl::VectorView(test_images_view.Width(), test_images_view[i]);
		testPred << NN::maxIndex(network.CalculateOutputs(vecView)) << std::endl;
	}
	testPred.close();

	for (size_t i = 0; i < train_images_view.Height(); i++)
	{
		auto vecView = VecAl::VectorView(train_images_view.Width(), train_images_view[i]);
		trainPred << NN::maxIndex(network.CalculateOutputs(vecView)) << std::endl;
	}
	trainPred.close();

	return 0;
}