package com.deeplearning.runners;

import static com.deeplearning.loader.image.ImageSetReader.getDataSetIteratorScaler;
import static com.deeplearning.loader.image.ImageSetReader.getImageRecordReader;
import static com.deeplearning.loader.image.ImageSetReader.getSplitFileSet;
import static com.deeplearning.utils.Constants.BATCH_SIZE;
import static com.deeplearning.utils.Constants.CHANNELS;
import static com.deeplearning.utils.Constants.EPOCH;
import static com.deeplearning.utils.Constants.HEIGTH;
import static com.deeplearning.utils.Constants.OUTPUT_NUM;
import static com.deeplearning.utils.Constants.RAND_NUM_GEN;
import static com.deeplearning.utils.Constants.TEST_PATH;
import static com.deeplearning.utils.Constants.TRAINING_PATH;
import static com.deeplearning.utils.Constants.WIDTH;

import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.deeplearning.network.factory.NetworkForImageFactory;
import com.deeplearning.server.ConfigureListenerServerUI;

public class NeuronalNetworkTrainer {
	
	private static Logger logger = LogManager.getLogger(NeuronalNetworkTrainer.class);
	private MultiLayerNetwork network;

	public void train() {
		
		logger.info("Iniciando aplicacion...");
		logger.info("Identificacion de caracteres numericos mediante redes neuronales...");
		logger.info("Generando fileSplits para la aplicacion...");

		try {

			logger.info("Configurando la red neuronal...");					
			network = NetworkForImageFactory.buildNeuronalNetworkWithListener();

			logger.info("Configurando servidor de visualizacion...");
			ConfigureListenerServerUI.configureListenerServerForUI(network);
			
			FileSplit splitTrainData = getSplitFileSet(TRAINING_PATH, RAND_NUM_GEN);
			ImageRecordReader recordReader = getImageRecordReader(HEIGTH, WIDTH, CHANNELS);

			logger.info("Entrenando la red neuronal...");
			recordReader.initialize(splitTrainData);	
			DataSetIterator dataTrainIterator = getDataSetIteratorScaler(recordReader, BATCH_SIZE, OUTPUT_NUM);			
			network.fit(dataTrainIterator, EPOCH);

			logger.info("Leyendo datos para evaluar la red neuronal...");
			FileSplit splitTestData = getSplitFileSet(TEST_PATH, RAND_NUM_GEN);
			recordReader.reset();
			recordReader.initialize(splitTestData);

			logger.info("Evaluando la red neuronal");
			DataSetIterator dataTestIterator = getDataSetIteratorScaler(recordReader, BATCH_SIZE, OUTPUT_NUM);			
			Evaluation evaluation = new Evaluation(OUTPUT_NUM);

			while (dataTestIterator.hasNext()) {
				DataSet nextDataSet = dataTestIterator.next();
				INDArray output = network.output(nextDataSet.getFeatures());
				evaluation.eval(nextDataSet.getLabels(), output);
			}

			logger.info(evaluation.stats());

		} catch (IOException e) {
			logger.error(e.getMessage());
		}

		logger.info("Red neuronal lista para ser usada...");		
	}
	
	public MultiLayerNetwork getNetwork() {
		return this.network;
	}

}
