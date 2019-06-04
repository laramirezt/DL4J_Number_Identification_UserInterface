package com.deeplearning.loader.image;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public final class ImageSetReader {
	
	private static final int INDEX_LABEL = 1;
	
	private ImageSetReader() {} 
	
	public static DataSetIterator getDataSetIteratorScaler(ImageRecordReader recordReader, int batchSize, int outputLabel) {
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, INDEX_LABEL, outputLabel);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);
		return dataIter;
	}
	
	public static ImageRecordReader getImageRecordReader(long heigth, long width, long channels) {
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		ImageRecordReader recordReader = new ImageRecordReader(heigth, width, channels, labelMaker);
		return recordReader;
	}

	public static FileSplit getSplitFileSet(String filePath, Random randomNumGenerator) {
		File trainData = new File(filePath);
		return new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randomNumGenerator);
	}

}
