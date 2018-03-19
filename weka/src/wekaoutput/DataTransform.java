package wekaoutput;

import weka.core.Instance;
import weka.core.Instances;
/*
 * 将多标签数据集原有的类标签合成为一个label（字符串）
 * 方便后来的计算最近邻操作
 */
public class DataTransform {
	String classes=new String();
	public void delete(Instances data) {
		for(int i=0;i<data.numInstances();i++) {
			int index=72;       //第一个类标签位置
		for(int j=0;j<6;j++) {
			Instance s=data.instance(i);
 			int temp=(int)(data.instance(i).value(index++));
			classes+=temp;
		}
			double cla=Double.parseDouble(classes);
		data.instance(i).setValue(78, cla);   //将所有的实例添加一列新的String label
		classes="";
		cla=0;
	}
	}
}
