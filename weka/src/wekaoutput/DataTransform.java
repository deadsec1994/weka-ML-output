package wekaoutput;

import weka.core.Instance;
import weka.core.Instances;
/*
 * �����ǩ���ݼ�ԭ�е����ǩ�ϳ�Ϊһ��label���ַ�����
 * ��������ļ�������ڲ���
 */
public class DataTransform {
	String classes=new String();
	public void delete(Instances data) {
		for(int i=0;i<data.numInstances();i++) {
			int index=72;       //��һ�����ǩλ��
		for(int j=0;j<6;j++) {
			Instance s=data.instance(i);
 			int temp=(int)(data.instance(i).value(index++));
			classes+=temp;
		}
			double cla=Double.parseDouble(classes);
		data.instance(i).setValue(78, cla);   //�����е�ʵ�����һ���µ�String label
		classes="";
		cla=0;
	}
	}
}
