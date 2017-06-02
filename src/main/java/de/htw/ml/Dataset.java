package de.htw.ml;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import org.jblas.FloatMatrix;

import com.sun.xml.internal.bind.v2.model.util.ArrayInfoUtil;

public class Dataset {
	
	protected Random rnd = new Random(7);
	
	protected FloatMatrix xTrain;
	protected FloatMatrix yTrain;
	
	protected FloatMatrix xTest;
	protected FloatMatrix yTest;
	protected int testDataCount;
	
	protected int[] categories;
	protected int[] categorySizes;
	
	public Dataset() throws IOException {
		
		int predictColumn = 15; // type of apartment
		FloatMatrix data = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		
		// Liste mit allen Kategorien die es in der predictColumn gibt
		final FloatMatrix outputData = data.getColumn(predictColumn);
		categories = IntStream.range(0, outputData.rows).map(idx -> (int)outputData.data[idx]).distinct().sorted().toArray();
		categorySizes = IntStream.of(categories).map(v -> (int)outputData.eq(v).sum()).toArray();
		System.out.println("The unique values of y are "+ Arrays.toString(categories)+" and there number of occurrences are "+Arrays.toString(categorySizes));

		// Array mit allen Zeilen die nicht predictColumn sind
		int[] xColumns = IntStream.range(0, data.columns).filter(value -> value != predictColumn).toArray();


		// normalisiere die Datensets
		FloatMatrix xNorm = getNormedXData(predictColumn, data);		
		
		// erstelle ein Trainings- und ein Testset mit 90% und 10% aller Daten
		testDataCount = data.getRows()/10; // 10% Testset
		FloatMatrix[] sets  = devideDataSet(predictColumn, 0.5f, 0.9f, 0.1f, xNorm);
		this.xTrain = sets[1].getColumns(xColumns);
		this.xTest = sets[0].getColumns(xColumns);
		this.yTrain = sets[1].getColumn(predictColumn);
		this.yTest= sets[0].getColumn(predictColumn);
		System.out.println("Use "+(testDataCount/categories.length)+" elements per category as test data.\n");
		
	}
	
	private static FloatMatrix getNormedXData(int y, FloatMatrix cars) {
		FloatMatrix normedXData = new FloatMatrix(cars.rows, cars.columns);
		for (int i = 0; i < cars.columns; i++) {
			FloatMatrix normed =  i!=y ? Reggression.norm(cars.getColumn(i)): cars.getColumn(y);
			normedXData.putColumn(i,normed);
		}
		return normedXData;
	}
	
private static FloatMatrix[] devideDataSet(int column, float threshhold,float amountGroupOne, float amountGroupTwo, FloatMatrix xData) {
		
		FloatMatrix[] devidedData = new FloatMatrix[2];
		
		int maxAmount = (int) (xData.rows*(amountGroupOne+ amountGroupTwo));
		
		FloatMatrix testData = new FloatMatrix(maxAmount, xData.columns);
		
		int maxGroupAmount1 = (int) (xData.rows * amountGroupOne);
		int maxGroupAmount2 = (int) (xData.rows * amountGroupTwo);
		
		int group1Counter = 0;
		int group2Counter = 0;
		int added = 0;
		int[] addedIndeces = new int[maxAmount];
		FloatMatrix testColumn = xData.getColumn(column);	
		for (int i = 0; i < xData.rows; i++) {
			if(testColumn.get(i) < threshhold && group1Counter <= maxGroupAmount1){
				testData.putRow(added, xData.getRow(i));
				group1Counter++;
				addedIndeces[added]  = i;
				added++;
			}else if(testColumn.get(i) >= threshhold && group2Counter <= maxGroupAmount2){
				testData.putRow(added, xData.getRow(i));
				group2Counter++;
				addedIndeces[added]  = i;
				added++;	
			}
			if(added == maxAmount) break;
		}
		devidedData[0] = testData;
		FloatMatrix trainingData = new FloatMatrix(xData.rows - maxAmount, xData.columns);
		
		int[] trainingRows = getSkippedIndeces(addedIndeces, xData.rows);
		trainingData = xData.getRows(trainingRows);
		devidedData[1] = trainingData;
		return devidedData;
	}
	
	private static int[] getSkippedIndeces(int[] skip, int columns) {
		int[] indices = new int[columns-1];
		int c = 0;
		for (int i = 0; i < columns; i++) {
			final int j = i;
			boolean toSkip = IntStream.of(skip).anyMatch(x -> x == j);
			if (!toSkip) {
				indices[c] = i;
				c++;
			}
		}
		return indices;
	}

	public FloatMatrix getXTrain() {
		return xTrain;
	}

	public FloatMatrix getYTrain() {
		return yTrain;
	}

	public FloatMatrix getXTest() {
		return xTest;
	}

	public FloatMatrix getYTest() {
		return yTest;
	}

	public int[] getCategories() {
		return categories;
	}

	/**
	 * Bereitet die Trainingsdaten vor. Das Set hat genauso viele Datenpunkte mit gewünschten Kategorie 
	 * wie auch Datenpunkte mit einer anderen Kategorie. Alle Y-Daten sind aber binariziert:
	 *  - gewünschten Kategorie = 1
	 *  - andere Kategorien = 0
	 * 
	 * @param categoryIndex
	 * @return {x Matrix,y Matrix}
	 */
	public FloatMatrix[] createTrainingsSet(int categoryIndex) {
		int category = categories[categoryIndex];
		int trainingsCategorySize = categorySizes[categoryIndex] - (testDataCount/categories.length);
		
		// TODO Finde alle Indizies von Zeilen in der die Kategorie vorkommt. 
		// Suche 'genauso' viele Zeilen mit einer anderen Kategorie. Entferne 
		// eventuell einige der gefunden Indizies, so dass die Anzahl auch 
		// wirklich gleich ist.
		int[] rowCatIndizies = IntStream.range(0, yTrain.rows).filter(value -> value == category).toArray();
		int[] rowOtherIndizies = IntStream.range(0, yTrain.rows).filter(value -> value != category).toArray();
		
		if(rowCatIndizies.length != rowOtherIndizies.length){
			rowOtherIndizies = Arrays.copyOfRange(rowOtherIndizies, 0, rowCatIndizies.length);
		}
		
		int[] rowIndizies = new int[rowCatIndizies.length + rowOtherIndizies.length];
		System.arraycopy(rowCatIndizies, 0, rowIndizies, 0, rowCatIndizies.length);
		System.arraycopy(rowOtherIndizies, 0, rowIndizies, rowCatIndizies.length, rowOtherIndizies.length);
		
		// besorge die gewünschten Datenpunkte und binarisiere die Y-Werte
		return new FloatMatrix[] { xTrain.getRows(rowIndizies), yTrain.getRows(rowIndizies).eq(category) };
	}

	/**
	 * Bereitet das Testset vor. Binariziert die Kategorien:
	 *  - gewünschten Kategorie = 1
	 *  - andere Kategorien = 0
	 *  
	 * @param categoryIndex
	 * @return {x,y}
	 */
	public FloatMatrix[] createTestSet(int categoryIndex) {
		return new FloatMatrix[] { xTest, yTest.eq(categories[categoryIndex]) };
	}
}
