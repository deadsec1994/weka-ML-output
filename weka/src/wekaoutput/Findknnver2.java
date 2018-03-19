package wekaoutput;

import java.io.BufferedReader;
import java.io.File;
import java.text.DecimalFormat;
import java.io.*;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Findknnver2 {

	public static void main(String[] args) throws Exception {
		// TODO 自动生成的方法存根
		Random random = new Random(1);
		String[] FileName={"emotions"};
		
		
		String inputfile = "G:\\mulan\\data\\testData\\"+FileName[0]+"-train.arff";
		String inputfile2= "G:\\mulan\\data\\testData\\"+FileName[0]+"-test.arff";
		FileReader fr = new FileReader(inputfile);
		FileReader tr = new FileReader(inputfile2);
		BufferedReader br = new BufferedReader(fr);
		BufferedReader cr = new BufferedReader(tr);
		
		Instances trdata = new Instances(br);
		Instances testdata = new Instances(cr);
		
		DataTransform trs=new DataTransform();
		Attribute classes=new Attribute("classes",false);
		DecimalFormat df=new DecimalFormat("000000");
		
		trdata.insertAttributeAt(classes , trdata.numAttributes());
		testdata.insertAttributeAt(classes , testdata.numAttributes());
		
		
		trs.delete(trdata);
		trs.delete(testdata);
		//删除原有类标签,index第一个类标签位置，每删一个类标签整体向前移动一位
		for(int d1=0,index=72;d1<6;d1++) {
			trdata.deleteAttributeAt(index);
		}
		for(int d2=0,index=72;d2<6;d2++) {
			testdata.deleteAttributeAt(index);
		}
		
		testdata.setClassIndex(testdata.numAttributes() - 1);
		trdata.setClassIndex(trdata.numAttributes() - 1);
		System.out.println("datanum:"+trdata.numInstances());
		System.out.println("classnum:"+trdata.numClasses());	
		double[] neighborindice = null;

				
			MyIbk a = new MyIbk();
			//String[] options;
			//a.setOptions(options);
			// 写出
			File trainout = new File("D:\\knnsearch\\train.txt");
			File testout = new File("D:\\knnsearch\\test.txt");
			File neighborout=new File("D:\\knnsearch\\trainclass.txt");
			File testclassout=new File("D:\\knnsearch\\testclass.txt");
			PrintStream out = new PrintStream(trainout);
			PrintStream out2 = new PrintStream(testout);
			PrintStream out3 = new PrintStream(neighborout);
			PrintStream out4 = new PrintStream(testclassout);
			double[] classValuetr=new double[trdata.numInstances()];
			for (int j = 0; j < trdata.numInstances(); j++) {
				Instance curInstance = trdata.get(j);
				
				trdata.delete(j);
				a.setKNN(3);
				a.buildClassifier(trdata);
				
				neighborindice = a.distributionForInstance(curInstance,j);
				trdata.add(j, curInstance);
				classValuetr[j] = trdata.instance(j).classValue();
				int index=j+1;
				for (int k = 0; k < neighborindice.length; k++) {  //输出训练集的邻居
					out.println(index+ ","+(int)neighborindice[k]);
				}
//				out.println(index+ ","+(int)(curInstance.classValue()+train.numInstances()+2));  
				//+2因为在net中train.numInstances()+1那一行作为测试实例输入位置

			}
			for(int j1=0;j1<classValuetr.length;j1++){
				int classval=(int)(classValuetr[j1]);
				String str2=df.format(classval);
				StringBuffer temp=new StringBuffer(str2);
				for(int j2=1;j2<10;j2+=2) {
					temp.insert(j2, ',');
				}
				//System.out.println(temp);
				out3.println(temp);
			}

			
			out.close();
            a.buildClassifier(trdata);

			double[] classValue=new double[testdata.numInstances()];

			for (int k = 0; k < testdata.numInstances(); k++) {  


				classValue[k] = testdata.instance(k).classValue();
			
				neighborindice = a.distributionForInstance(testdata.instance(k),trdata.numInstances());

				for (int i1 = 0; i1 < neighborindice.length; i1++) {

					out2.print((int)(neighborindice[i1])+",");
					
				}
				out2.print("\r\n");
				

				
			}
			//输出测试集的类标签
			for(int i2=0;i2<classValue.length;i2++){
				int classval=(int)(classValue[i2]);
				String str2=df.format(classval);
				out4.println(str2);
			}
			out2.close();
			out3.close();
			out4.close();


	}
}
