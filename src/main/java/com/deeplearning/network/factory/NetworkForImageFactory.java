package com.deeplearning.network.factory;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.deeplearning.utils.Constants;

public final class NetworkForImageFactory {

	private NetworkForImageFactory() {}

	public static MultiLayerNetwork buildNeuronalNetwork() {
		return new MultiLayerNetwork(generateConfiguration());
	}
	
	public static MultiLayerNetwork buildNeuronalNetworkWithListener() {
		MultiLayerNetwork network = new MultiLayerNetwork(generateConfiguration());
		network.setListeners(new ScoreIterationListener(100));
		return network;
	}

	private static MultiLayerConfiguration generateConfiguration() {
		return new NeuralNetConfiguration.Builder()
				.seed(Constants.SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Nesterovs(Constants.LEARNING_RATE, Constants.MOMENTUM))
				.weightInit(WeightInit.XAVIER)
				.l2(Constants.L2)
				.list()
				.layer(0, new DenseLayer.Builder()
						.activation(Activation.RELU)
						.nIn(Constants.LAYER_0_INPUT)
						.nOut(Constants.LAYER_0_OUTPUT)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
						.nIn(Constants.LAYER_1_INPUT)
						.nOut(Constants.LAYER_1_OUTPUT)
						.build())
				.setInputType(InputType.convolutional(Constants.HEIGTH, Constants.WIDTH, Constants.CHANNELS))
				.build();
	}

}
