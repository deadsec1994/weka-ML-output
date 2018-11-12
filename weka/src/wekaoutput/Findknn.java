package wekaoutput;

import java.io.BufferedReader;
import java.io.File;
import java.text.DecimalFormat;
import java.io.*;
import java.util.Random;

import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class Findknn {

	public static void main(String[] args) throws Exception {
		// TODO 自动生成的方法存根
		Random random = new Random(1);
		String[] FileName={"traditional","birds","emotions","yeast2","scene","CAL500","Corel5k",
				"medical","Atrs1","Business1","Computers1","Education1","Entertainment1","Health1",
				"Recreation1","Reference1","Science1","Social1","Society1"};
		
		String inputfile = "C:\\Users\\wwwcu\\Desktop\\data\\"+FileName[4]+".arff";
		FileReader fr = new FileReader(inputfile);
		BufferedReader br = new BufferedReader(fr);
		Instances data = new Instances(br);
		DataTransform tra=new DataTransform();
		Attribute classes=new Attribute("classes",true);
		int trainSize = (int) Math.round(data.numInstances() * 66 / 100); //66%作为训练集
	    int testSize = data.numInstances() - trainSize;
		
		 ArffSaver saver1 = new ArffSaver();
		 ArffSaver saver2 = new ArffSaver();
		 
		 saver1.setFile(new File("C:\\Users\\wwwcu\\Desktop\\data\\"+FileName[7]+"-train.arff"));
		 saver2.setFile(new File("C:\\Users\\wwwcu\\Desktop\\data\\"+FileName[7]+"-test.arff"));

	     
	     Instances trainfout = new Instances(data, 0, trainSize);
	     Instances testfout = new Instances(data, trainSize, testSize);
	     
	     saver1.setInstances(trainfout);
		 saver2.setInstances(testfout);
	     saver1.writeBatch();
	     saver2.writeBatch();
		
		int first_Class_loc = 294; //手动改10，260,
		int numofClass = 6;  //手动改21,19,6,14
		System.out.println(data.numAttributes());
		
		data.insertAttributeAt(classes , data.numAttributes());
		
		
		tra.delete(data,first_Class_loc,numofClass);
		
		//删除原有类标签,index第一个类标签位置，每删一个类标签整体向前移动一位
		for(int d1=0,index=first_Class_loc;d1<numofClass;d1++) {
              data.deleteAttributeAt(index);
		}
		
		data.setClassIndex(data.numAttributes() - 1);
		System.out.println("datanum:"+data.numInstances());
		System.out.println("classnum:"+data.numClasses());	
		System.out.println(data.instance(0).attribute(first_Class_loc));
		double[] neighborindice = null;
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(10);
		}


//		for (int i = 0; i < 10; i++) {// 二维数组
//			Instances train = data.trainCV(10, i);
//			Instances test = data.testCV(10, i);
//			train.setClassIndex(train.numAttributes() - 1);
//			test.setClassIndex(test.numAttributes() - 1);
			
			
			
		Instances train = new Instances(data, 0, trainSize);
	    Instances test = new Instances(data, trainSize, testSize);
	       

			
			MyIbk a = new MyIbk();
			//String[] options;
			//a.setOptions(options);
			// 写出
//			File trainout = new File("D:\\knnsearch\\c=" + i + "\\trainout.txt");
			File trainout = new File("D:\\knnsearch\\trainout.txt");
			File testout = new File("D:\\knnsearch\\testout.txt");
			File trainclass=new File("D:\\knnsearch\\trainclass.txt");
			File testclass=new File("D:\\knnsearch\\testclass.txt");
			PrintStream out = new PrintStream(trainout);
			PrintStream out2 = new PrintStream(testout);
			PrintStream out3 = new PrintStream(trainclass);
			PrintStream out4 = new PrintStream(testclass);
						
			
			for (int j1=0;j1<train.numInstances();j1++) {	
				double classval=train.instance(j1).classValue();
				String cla = train.instance(j1).attribute(first_Class_loc).value((int)classval);
				//String str2=df.format(classval);
				StringBuffer temp=new StringBuffer(cla);
				for(int j11=1;j11<numofClass*2;j11+=2) {
					temp.insert(j11, ',');
				}
				out.println(train.instance(j1));
				out3.println(temp);
			}

			for (int j2=0;j2<test.numInstances();j2++) {	
				double classval=test.instance(j2).classValue();
				String cla = train.instance(j2).attribute(first_Class_loc).value((int)classval);
				StringBuffer temp=new StringBuffer(cla);
				for(int j22=1;j22<numofClass*2;j22+=2) {
					temp.insert(j22, ',');
				}
				out2.println(test.instance(j2));
				out4.println(temp);
			}
			out.close();
			out2.close();
			out3.close();
			out4.close();
			}

		}

//	}
