package com.deeplearning.loader.image;

import java.io.File;
import java.io.IOException;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public final class SingleImage {
	
	private SingleImage() {}
	
	public static INDArray readNormalizeImage(File fileImage, long height, long width, long channels) throws IOException {
		NativeImageLoader imageLoader = new NativeImageLoader(height, width, channels);
	    INDArray imageArray = imageLoader.asMatrix(fileImage);
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	    scaler.transform(imageArray);
		return imageArray;		
	}

}
