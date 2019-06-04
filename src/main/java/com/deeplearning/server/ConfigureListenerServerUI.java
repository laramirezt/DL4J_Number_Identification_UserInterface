package com.deeplearning.server;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

public final class ConfigureListenerServerUI {
	
	private ConfigureListenerServerUI() {}
	
	public static void configureListenerServerForUI(MultiLayerNetwork neuronalNetwork) {
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		neuronalNetwork.setListeners(new StatsListener(statsStorage));
	}

}
