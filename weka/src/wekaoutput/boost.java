package wekaoutput;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;  
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;  

public class boost {
	public static void main(String[] args) throws Exception {  

        /* 
         * 定义评价指标
         */
        double AvgCorrect = 0;
		double Precision = 0; 
		double Recall = 0;   
		double F_Measure = 0;
		int numofCla = 21;
		
		
		double AvgCorrect2 = 0;
		double Precision2 = 0; 
		double Recall2 = 0;
		double F_Measure2 = 0;
		
		
		File txtoutput = new File("D:\\result.txt");
		FileOutputStream  fos=new FileOutputStream(txtoutput,true);
		PrintStream ps = new PrintStream(fos);
		
		
//		for(int i = 0; i < 10; i++) {
	        DataSource datasource1 =new DataSource("D:\\knnsearch\\train_out.csv");  
	        DataSource datasource2 =new DataSource("D:\\knnsearch\\test_old.csv");
	        Instances data = datasource1.getDataSet();
	        Instances data2 = datasource2.getDataSet();
	        NumericToNominal filter=new NumericToNominal();
	        filter.setInputFormat(data);
	        String options[]=new String[2];
	        options[0]="-R";
//	        options[1]="first,last"; 
	        options[1]="6-6";           //将类标签变为normal型，24为类标签索引位置
	        filter.setOptions(options);
	        Instances train = Filter.useFilter(data, filter);
	        Instances test = Filter.useFilter(data2, filter);
	        if(train.classIndex()==-1)  
	        	train.setClassIndex(train.numAttributes()-1);  
	        if(test.classIndex()==-1)  
	        	test.setClassIndex(test.numAttributes()-1); 
			//System.out.println(test.instance(i).classValue());
	        AdaBoostM1 classifier = new AdaBoostM1();
	        Bagging bagclassifier = new Bagging();
	        J48 baseClassifier = new J48();
	        
	        Measure m = new Measure();
	        DataTransform df = new DataTransform();
	        
	        classifier.setClassifier( baseClassifier ); 
	        bagclassifier.setClassifier(baseClassifier);
	        bagclassifier.buildClassifier(train);
	        classifier.buildClassifier( train );
	        
	        
	        Evaluation testingEvaluation = new Evaluation(train);
	        Evaluation testingEvaluation2 = new Evaluation(train);
        
	        double predictions[] = new double[numofCla];
	        double Real[] = new double[numofCla];
	        
	        int count = 0;
	        m.reset();
	        for(int i = 0;i<test.numInstances();i++) {

	        	predictions[count] = classifier.classifyInstance(test.instance(i));
	        	Real[count] = test.instance(i).classValue();
	        	if(count == numofCla-1) {
		        	boolean[] predict_label = df.toBool(predictions);
		        	boolean[] real_label = df.toBool(Real);
	        		count = 0;
	        		m.Accuracy(predict_label, real_label);
	        		m.Recall(predict_label, real_label);
	        		m.Precision(predict_label, real_label);
	        		m.HammingLoss(predict_label, real_label);
	        	}else
	        		count++;
	        }
	        
	        AvgCorrect = m.getValue("-A");
	        Precision = m.getValue("-P");
	        Recall = m.getValue("-R");
	        double HammingLoss1 = m.getValue("-H");
	        m.reset();
	        
	        for(int i = 0;i<test.numInstances();i++) {

	        	predictions[count] = bagclassifier.classifyInstance(test.instance(i));
	        	Real[count] = test.instance(i).classValue();
	        	if(count == numofCla-1) {
		        	boolean[] predict_label = df.toBool(predictions);
		        	boolean[] real_label = df.toBool(Real);
	        		count = 0;
	        		m.Accuracy(predict_label, real_label);
	        		m.Recall(predict_label, real_label);
	        		m.Precision(predict_label, real_label);
	        		m.HammingLoss(predict_label, real_label);
	        	}else
	        		count++;
	        }
	        
	        AvgCorrect2 = m.getValue("-A");
	        Precision2 = m.getValue("-P");
	        Recall2 = m.getValue("-R");
	        double HammingLoss2 = m.getValue("-H");
	        
	        F_Measure = (2*Precision*Recall)/(Precision+Recall);
			F_Measure2 = (2*Precision2*Recall2)/(Precision2+Recall2);
	        
	        
	        
	        
//	        double HammingLoss1 = testingEvaluation.HammingLoss(predictions1, test);
//	        double HammingLoss2 = testingEvaluation2.HammingLoss(predictions2, test);
//	        AvgCorrect += testingEvaluation.pctCorrect();
//	        Precision += testingEvaluation.weightedPrecision();
//	        Recall += testingEvaluation.weightedRecall();
////	        F_Measure +=testingEvaluation.weightedFMeasure();
//	        
//	        AvgCorrect2 += testingEvaluation2.pctCorrect();
//	        Precision2 += testingEvaluation2.weightedPrecision();
//	        Recall2 += testingEvaluation2.weightedRecall();
	        
	        
	        
	        
	        
	        

	        System.out.println(testingEvaluation.toMatrixString());
//	        System.out.println("end"+i);

//		}
		
		
		System.out.println("BoostAvgCorrect ："+ AvgCorrect + "\n" + "BoostPrecision:" + 
							Precision + "\n" + "BoostRecall: "+ Recall +"\n" + "BoostF-measure: " 
							+ F_Measure + "\n" + "BoostHammingLoss: "+HammingLoss1 
							);
		System.out.println("BaggingAvgCorrect："+ AvgCorrect2 + "\n" + "BaggingPrecision:" + 
				Precision2 + "\n" + "BaggingRecall: "+ Recall2 +"\n" + "BaggingF-measure: " 
				+ F_Measure2 + "\n" + "BaggingHammingLoss: " +HammingLoss2);
//		ps.append("Arts1:\r\n"
//				 + "BoostAvgCorrect:"+AvgCorrect+"     BaggingAvgCorrect:" + AvgCorrect2 + "\r\n"
//				 + "BoostPrecision:" + Precision + "     BaggingPrecision:" + Precision2 + "\r\n"
//				 + "BoostRecall:" + Recall + "     BaggingRecall:" + Recall2 + "\r\n" 
//				 + "BoostF-measure:" + F_Measure + "     BaggingF-measure:" + F_Measure2 );
//		ps.close();
	}	
}

