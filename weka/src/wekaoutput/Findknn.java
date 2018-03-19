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

public class Findknn {

	public static void main(String[] args) throws Exception {
		// TODO 自动生成的方法存根
		Random random = new Random(1);
		String[] FileName={"emotions","seismic-bumps","credit-g","diabetes","glass","iris","QualBank","ThoraricSurgery",
				"vertebral","vote","breast-cancer","Phishing data","sponge","car","cylinder-bands","Diabetic Retinopathy Debrecen","mfeat-morphological","wilt","page-blocks","hypothyroid","arrhythmia","PC5",
				"QualBank"};
		
		
		String inputfile = "C:\\Users\\wwwcu\\Desktop\\knnsearch\\uci\\"+FileName[0]+".arff";
		FileReader fr = new FileReader(inputfile);
		BufferedReader br = new BufferedReader(fr);
		Instances data = new Instances(br);
		DataTransform tra=new DataTransform();
		Attribute classes=new Attribute("classes",false);
		DecimalFormat df=new DecimalFormat("000000");
		
		data.insertAttributeAt(classes , data.numAttributes());
		
		tra.delete(data);
		//删除原有类标签,index第一个类标签位置，每删一个类标签整体向前移动一位
		for(int d1=0,index=72;d1<6;d1++) {
              data.deleteAttributeAt(index);
		}
		
		data.setClassIndex(data.numAttributes() - 1);
		System.out.println("datanum:"+data.numInstances());
		System.out.println("classnum:"+data.numClasses());	
		double[] neighborindice = null;
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(10);
		}
		
//		data.setClassIndex(data.numAttributes() - 1);
//		EuclideanDistance dis=new EuclideanDistance(data);
//		dis.setDontNormalize(true);
//		for(int i=0;i<data.numInstances();i++){
//			for(int j=i+1;j<data.numInstances();j++){
//				
//				System.out.println("(" + i + "."+j+ ")="+ dis.distance(data.instance(i), data.instance(j)));
//			}
//		}

		for (int i = 0; i < 10; i++) {// 二维数组
			Instances train = data.trainCV(10, i, random);
			Instances test = data.testCV(10, i);
//			train.setClassIndex(train.numAttributes() - 1);
//			test.setClassIndex(test.numAttributes() - 1);
			
			
			MyIbk a = new MyIbk();
			//String[] options;
			//a.setOptions(options);
			// 写出
			File trainout = new File("D:\\knnsearch\\KnnSearch" + i + ".txt");
			File testout = new File("D:\\knnsearch\\test" + i + ".txt");
			File neighborout=new File("D:\\knnsearch\\neighbor" + i + ".txt");
			File testclassout=new File("D:\\knnsearch\\testclass" + i + ".txt");
			PrintStream out = new PrintStream(trainout);
			PrintStream out2 = new PrintStream(testout);
			PrintStream out3 = new PrintStream(neighborout);
			PrintStream out4 = new PrintStream(testclassout);
			double[] classValuetr=new double[train.numInstances()];
			for (int j = 0; j < train.numInstances(); j++) {
				Instance curInstance = train.get(j);
				
				train.delete(j);
				a.setKNN(3);
				a.buildClassifier(train);
				// built
				//System.out.println("cur: " + i);
				//System.out.println("cur: " + j);
				neighborindice = a.distributionForInstance(curInstance,j);
				train.add(j, curInstance);
				classValuetr[j] = train.instance(j).classValue();
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
				out3.println(str2);
			}

			
			out.close();
            a.buildClassifier(train);
			//Instances test = data.testCV(10, i);
            int temp=test.numInstances();
			double[] classValue=new double[test.numInstances()];
			//double[] distribution=new double[train.numClasses()];
			for (int k = 0; k < test.numInstances(); k++) {  

				//test.setClassIndex(test.numAttributes() - 1);
				classValue[k] = test.instance(k).classValue();
			
				neighborindice = a.distributionForInstance(test.instance(k),train.numInstances());
				//distribution=a.distributionForInstance(test.instance(k));
				for (int i1 = 0; i1 < neighborindice.length; i1++) {
//					out2.println(index+","+(int) (neighborindice[i1]));
					out2.print((int)(neighborindice[i1])+",");
				}
				out2.print("\r\n");
				
//				for(int i2=0;i2<distribution.length;i2++){
//					out3.print(distribution[i2]+",");
//				}
				
				
				
				//获取测试样例邻居的类标签
				
//				for(int i2=0;i2<neighborindice.length; i2++){
//					Instance temp=train.instance((int) neighborindice[i2]-1);
//					double neghborValue= temp.classValue();
//					out3.print(neghborValue+",");
//				}
				
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
}
