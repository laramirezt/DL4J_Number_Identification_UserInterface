package com.deeplearning.utils;

import java.util.Random;

public final class Constants {
	
	private Constants() {}
	
	// Paths.
    public static final String PARENT_PATH = "C:/Users/user/Desktop/mnist_png";
	public static final String TRAINING_PATH = PARENT_PATH + "/training";
	public static final String TEST_PATH = PARENT_PATH + "/testing";
	
	// Data input.
	public static final int HEIGTH = 28;
	public static final int WIDTH = 28;
	public static final int CHANNELS = 1;
	public static final int RNGSEED = 123;	
	public static final int BATCH_SIZE = 128;
	
	// Neuronal Network.
	public static final int OUTPUT_NUM = 10;
	public static final int EPOCH = 1;
	public static final Random RAND_NUM_GEN = new Random(RNGSEED);
	public static final long SEED = 2345;
	public static final double L2 = 1e-4;
	public static final double LEARNING_RATE = 0.05;
	public static final double MOMENTUM = 0.9;
	public static final int LAYER_0_INPUT = HEIGTH * WIDTH;
	public static final int LAYER_0_OUTPUT = 100;
	public static final int LAYER_1_INPUT = 100;
	public static final int LAYER_1_OUTPUT = OUTPUT_NUM;
	
}
