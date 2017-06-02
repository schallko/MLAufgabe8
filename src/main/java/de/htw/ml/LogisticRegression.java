package de.htw.ml;

import org.jblas.FloatMatrix;

public class LogisticRegression {
	
	protected int trainingIterations;
	protected float learnRate;
	protected float[] predictionRates;
	protected float[] trainErrors;	
	
	public LogisticRegression(int trainingIterations, float learnRate) {
		this.trainingIterations = trainingIterations;
		this.learnRate = learnRate;
	}

	public FloatMatrix train(FloatMatrix xTest, FloatMatrix yTest, FloatMatrix xTrain, FloatMatrix yTrain) {
		this.predictionRates = new float[trainingIterations];
		this.trainErrors = new float[trainingIterations];
		
		// initializiere die Gewichte
		org.jblas.util.Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(xTrain.getColumns(), 1);
		
		// aktueller Trainingsfehler
		trainErrors[0] = cost(predict(xTrain, theta), yTrain);

		// beste kombination an Gewichten
		FloatMatrix bestTheta = theta.dup();
		float bestPredictionRate = 0;
		
		// training
		for (int iteration = 0; iteration < trainingIterations; iteration++) {
			// TODO Training für die logistische Regression		
		}
		
		return bestTheta;
	}

	/**
	 * Berechnet eine Prediction für die Eingangsdaten X und den aktuellen Gewichten theta
	 * 
	 * @param x
	 * @param theta
	 * @return
	 */
	public static FloatMatrix predict(FloatMatrix x, FloatMatrix theta) {
		// TODO Auto-generated method stub
		return null;
	}
		
	/**
	 * Berechnet den Trainingsfehler mit der logistischen Kostenfunktion oder den RMSE aus.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float cost(FloatMatrix prediction, FloatMatrix y) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Berechnet zwischen der Prediktion und den Wunschergebnis Y eine Prediktionsrate aus.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float predictionRate(FloatMatrix prediction, FloatMatrix y) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Prediction Rates vom letzten Training
	 * 
	 * @return
	 */
	public float[] getLastPredictionRates() {
		return predictionRates;
	}
	
	/**
	 * Error Rates vom letzten Training
	 * 
	 * @return
	 */
	public float[] getLastTrainError() {
		return trainErrors;
	}
	
	/**
	 * Ersetzt die Werte in der Input Matrix mit der sigmoid Variante.
	 * 
	 * @param input
	 * @return
	 */
	public static FloatMatrix sigmoidi(FloatMatrix input) {
		for (int i = 0; i < input.data.length; i++)
			input.data[i] = (float) (1. / ( 1. + Math.exp(-input.data[i]) ));
		return input;
	}
}