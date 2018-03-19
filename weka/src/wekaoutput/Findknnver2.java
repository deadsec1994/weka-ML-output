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
	
	
	//ɾ���˲��Լ����������Ϊleave_one_out

	public static void main(String[] args) throws Exception {
		// TODO �Զ����ɵķ������
		Random random = new Random(1);
		String[] FileName={"emotions"};
		
		
		String inputfile = "G:\\mulan\\data\\"+FileName[0]+".arff";
		FileReader fr = new FileReader(inputfile);

		BufferedReader br = new BufferedReader(fr);

		
		Instances trdata = new Instances(br);

		
		DataTransform trs=new DataTransform();
		Attribute classes=new Attribute("classes",false);
		DecimalFormat df=new DecimalFormat("000000");
		
		trdata.insertAttributeAt(classes , trdata.numAttributes());

		
		
		trs.delete(trdata);

		//ɾ��ԭ�����ǩ,index��һ�����ǩλ�ã�ÿɾһ�����ǩ������ǰ�ƶ�һλ
		for(int d1=0,index=72;d1<6;d1++) {
			trdata.deleteAttributeAt(index);
		}

		
		trdata.setClassIndex(trdata.numAttributes() - 1);
		System.out.println("datanum:"+trdata.numInstances());
		System.out.println("classnum:"+trdata.numClasses());	
		double[] neighborindice = null;

				
			MyIbk a = new MyIbk();
			//String[] options;
			//a.setOptions(options);
			// д��
			File trainout = new File("D:\\data\\train.txt");
			File neighborout=new File("D:\\data\\trainclass.txt");

			PrintStream out = new PrintStream(trainout);

			PrintStream out3 = new PrintStream(neighborout);

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
				for (int k = 0; k < neighborindice.length; k++) {  //���ѵ�������ھ�
					out.println(index+ ","+(int)neighborindice[k]);
				}

			}
			for(int j1=0;j1<classValuetr.length;j1++){
				int classval=(int)(classValuetr[j1]);
				String str2=df.format(classval);
				StringBuffer temp=new StringBuffer(str2);
				for(int j2=1;j2<10;j2+=2) {
					temp.insert(j2, ',');
				}

				out3.println(temp);
			}

			
			out.close();
			out3.close();
            a.buildClassifier(trdata);

			}

	}
