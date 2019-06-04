package com.deeplearning.main;

import java.io.File;
import java.io.IOException;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.deeplearning.loader.image.SingleImage;
import com.deeplearning.runners.NeuronalNetworkTrainer;
import static com.deeplearning.utils.Constants.CHANNELS;
import static com.deeplearning.utils.Constants.HEIGTH;
import static com.deeplearning.utils.Constants.WIDTH;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.effect.DropShadow;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

public class NumberRecognizerApp extends Application {

	private static Logger logger = LogManager.getLogger(NumberRecognizerApp.class);
	private Stage primaryStage;
	private Label numberResult;
	private MultiLayerNetwork network;

	public static void main(String[] args) {
		launch(args);		
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		this.primaryStage = primaryStage;
		logger.debug("Iniciando la interfaz de usuario");
		Scene primaryScene = new Scene(generateGraphicsComponents(), 400, 150);
		this.primaryStage.setScene(primaryScene);
		this.primaryStage.setX(100);
		this.primaryStage.setY(100);
		this.primaryStage.show();		
	}

	private Group generateGraphicsComponents() {
		Group root = new Group();
		numberResult = createGenericLabel("0", 320, 20, getlabelStyle());
		Label labelResult = createGenericLabel("Resultado", 310, 10, "");
		Button trainNetworkButton = createGenericButton("Entrenar Red", getButtonStyle("darkslateblue"), 10, 28);
		Button makePredictionButton = createGenericButton("Evaluar imagen", getButtonStyle("green"), 150, 28);
		setButtonActions(trainNetworkButton, makePredictionButton);
		root.getChildren().add(labelResult);
		root.getChildren().add(numberResult);
		root.getChildren().add(trainNetworkButton);
		root.getChildren().add(makePredictionButton);
		return root;
	}
	
	private Button createGenericButton(String label, String style, int xPos, int yPos) {
		Button genericButton = new Button();
		genericButton.setText(label);
		genericButton.setStyle(style);	
		genericButton.setLayoutX(xPos);
		genericButton.setLayoutY(yPos);
		setButtonEffects(genericButton);
		return genericButton;
	}
	
	private Label createGenericLabel(String text, int posX, int posY, String style) {
		Label genericLabel = new Label(text);
		genericLabel.setLayoutX(posX);
		genericLabel.setLayoutY(posY);
		genericLabel.setStyle(style);
		return genericLabel;		
	}
	
	private String getButtonStyle(String color) {
		return 	"-fx-background-color: " + color + "; " +
				"-fx-text-fill: white; " + 
				"-fx-padding: 40px 20px; " + 
				"-fx-background-radius: 15%; ";
	}
	
	private String getlabelStyle() {
		return "-fx-font-size: 80px; " +
			   "-fx-font-weight: bold;";
	}
	
	public void setButtonEffects(Button button){
		DropShadow shadow = new DropShadow();
        shadow.setColor(Color.YELLOW);      
        shadow.setRadius(10.0); 
        shadow.setSpread(0.9);
        button.addEventHandler(MouseEvent.MOUSE_ENTERED, (MouseEvent event) -> button.setEffect(shadow));    
        button.addEventHandler(MouseEvent.MOUSE_EXITED, (MouseEvent event) -> button.setEffect(null));       
    }
	
	public void setButtonActions(Button trainNetworkButton, Button makePredictionButton) {
		trainNetworkButton.setOnAction(getTrainnerButtonAction());
		makePredictionButton.setOnAction(getPredictorButtonAction());
	}
	
	public EventHandler<ActionEvent> getTrainnerButtonAction() {
		return event -> {
			NeuronalNetworkTrainer trainer = new NeuronalNetworkTrainer();
			trainer.train();
			network = trainer.getNetwork();
		};
	}
	
	public EventHandler<ActionEvent> getPredictorButtonAction() {
		return event -> {
			FileChooser imageSelector = createImageSelector();
            File image = imageSelector.showOpenDialog(primaryStage);
	        if (image != null) {
	            makeNumberPrediction(image);
	        }
		};
	}

	private FileChooser createImageSelector() {
		FileChooser imageSelector = new FileChooser();
		imageSelector.getExtensionFilters().addAll(
				new FileChooser.ExtensionFilter("All images", "*.*"), 
				new FileChooser.ExtensionFilter("JPG", "*.jpg"), 
				new FileChooser.ExtensionFilter("PNG", "*.png")
				);
		return imageSelector;
	}

	private void makeNumberPrediction(File image) {
		try {
			INDArray imageArray = SingleImage.readNormalizeImage(image, HEIGTH, WIDTH, CHANNELS);
			numberResult.setText(Integer.toString(network.predict(imageArray)[0]));
		} catch (IOException e) {
			logger.error(e.getMessage());
		}		
	}
	
}
