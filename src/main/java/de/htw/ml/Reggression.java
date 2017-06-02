package de.htw.ml;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class Reggression {



	
	public static float calcRmse(FloatMatrix yProxy, FloatMatrix yReal){
		float rmse = 0;
		float summ = MatrixFunctions.pow(yProxy.sub(yReal),2).sum();
		rmse = (float) Math.sqrt(summ/ yReal.length);
		return rmse;
	}
	
	public static FloatMatrix norm(FloatMatrix matrix){
		//norm = (x-min(x))/(max(x)-min(x));
		float min = matrix.min();
		float max = matrix.max();
		
		return matrix.subi(min).div(max - min);
	}
	
	public static FloatMatrix denorm(FloatMatrix matrix, FloatMatrix normedMatrix){
		
		//denorm = norm * (max(x) - min(x)) + min(x);
		return normedMatrix.mul((matrix.max() - matrix.min()) + matrix.min());
	}
	
	
	
	/**
	 * function linearMult = linearMultFun(M,V)
		linearMult = [rows(M)];
		  for i=1 : rows(M)
		    linearMult(i) = M(i,:) * V';
		  endfor
		endfunction
	 */
	
	public static FloatMatrix linearMult(FloatMatrix matrix, FloatMatrix vector){
		FloatMatrix result = new FloatMatrix(matrix.rows,1);
		for (int i = 0; i < matrix.rows; i++) {
			FloatMatrix multResult = matrix.getRow(i).mmul(vector);
			result.putRow(i, multResult);
		}
		return result;
	}
	
	/**
	 * function newTheta = costFun(xData, theta, alpha,yData)
		  hypothese = linearMultFun(xData,theta)';
		  diff = hypothese - yData;
		  desiredChange = linearMultFun(xData', diff');
		  damping = alpha/length(yData);
		  dampedValues = desiredChange * damping;
		  newTheta = theta - dampedValues;
		endfunction
	 */
	
	public static FloatMatrix newTheta(FloatMatrix xData, FloatMatrix oldTheta, float alpha,FloatMatrix yData){
		FloatMatrix hypothese = linearMult(xData, oldTheta);
		FloatMatrix diff = hypothese.sub(yData);
		FloatMatrix desiredChange = linearMult(xData.transpose(), diff);
		float damping = alpha/yData.length;
		FloatMatrix dampedValues = desiredChange.mul(damping);
		
		return oldTheta.sub(dampedValues);
	}
	
	public static FloatMatrix newThetaLog(FloatMatrix xData, FloatMatrix oldTheta, float alpha,FloatMatrix yData){
		FloatMatrix hypothese = getHypothese(xData, oldTheta);
		FloatMatrix diff = hypothese.sub(yData);
		FloatMatrix desiredChange = linearMult(xData.transpose(), diff);
		float damping = alpha/yData.length;
		FloatMatrix dampedValues = desiredChange.mul(damping);
		
		return oldTheta.sub(dampedValues);
	}

	public static FloatMatrix getHypothese(FloatMatrix xData, FloatMatrix oldTheta) {
		FloatMatrix hypothese = sigmuidFunction(linearMult(xData, oldTheta));
		return hypothese;
	}
	
	private static FloatMatrix sigmuidFunction(FloatMatrix matrix){
		// (1/(1+Math.pow(Math.E,z * -1)));
		
		for (int i = 0; i < matrix.length; i++) {
			matrix.put(i,(float) (1/(1+Math.pow(Math.E,matrix.get(i) * -1))));
		}
		 return matrix;
	}
	
	public static float getPredictionRate(FloatMatrix hyp, FloatMatrix yData, float threshold,FloatMatrix xData){
		for (int i = 0; i < hyp.length; i++) {
			hyp.put(i, hyp.get(i) > threshold ? 1 :0);
		}
		float error =  MatrixFunctions.abs(hyp.sub(yData)).sum();
		return (yData.rows - error)/yData.rows * 100;	
	}
	
	
	

}
