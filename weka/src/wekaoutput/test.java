package wekaoutput;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.plaf.synth.SynthSpinnerUI;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
public class test {

	public static void main(String[] args) throws Exception {
		IBk classifier = new IBk();
		classifier.setKNN(3);
		int numFolds =10;
		Random random=new Random(1);
//		String[] options={"-K3","-W0"};
		
		String[] options = {"-K", "3", "-W", "0","-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\""};
		classifier.setOptions(options);
		DataSource source = new DataSource("C:\\Users\\wwwcu\\Desktop\\knnsearch\\knnsearch\\iris.arff");
		Instances instances = source.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		Instances template = new Instances(instances, 0);
			Evaluation testingEvaluation =
					new Evaluation(new Instances(template, 0), null);
			testingEvaluation.setDiscardPredictions(false);
			testingEvaluation.toggleEvalMetrics(new ArrayList<String>());
			instances.randomize(random);
			if (instances.classAttribute().isNominal()) {
				instances.stratify(numFolds);
			}
			Object[] pre = new Object[0];
			for(int j = 0; j < numFolds; j++){
				Instances train = instances.trainCV(numFolds, j, random);
		    	Instances test = instances.testCV(numFolds, j);
		    	Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
		    	testingEvaluation.setPriors(train);
		    	copiedClassifier.buildClassifier(train);
//		    	Evaluation.evaluateModel(copiedClassifier, options);
		    	testingEvaluation.evaluateModel(copiedClassifier, test);
		    	
//			double curAcc = testingEvaluation.weightedTruePositiveRate();
//			double curAuc = testingEvaluation.weightedAreaUnderROC();
//			System.out.print("iris" + "\'s weighted true positive rate: " + curAcc);
		}
			double a=testingEvaluation.pctCorrect();
	    	System.out.println("acc"+a);
	}
}



