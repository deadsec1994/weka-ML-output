package wekaoutput;

import weka.core.Instances;

public class DataTransform {
	String classes=new String();
	/**
	 * @Description:
	 * �����ǩ���ݼ�ԭ�е����ǩ�ϳ�Ϊһ��label���ַ�����
	 * ��������ļ�������ڲ���
	 * @param data ��Ҫɾ�������ݼ�
	 * @param index ���ǩ��ʼλ��
	 * @param NumOfClass ���ǩ����
	 * @param numofAttr �����ǩλ��
	 * @return �ϲ����ǩ������ݼ�
	 * @author Cui
	 */
	public void delete(Instances data , int index , int NumOfClass ) {   
		int numofAttr = data.numAttributes()-1;
		for(int i=0;i<data.numInstances();i++) {
			int trindex = index;
			for(int j=0;j<NumOfClass;j++) {
 			int temp=(int)(data.instance(i).value(trindex++));
			classes+=temp;
			}
//		System.out.println(data.instance(i).value(279));
		//Double cla=Double.parseDouble(classes);
		data.instance(i).setValue(numofAttr, classes);   //�����һ�и��µ�labelֵ
		classes="";
		//cla=0;
		}
	}
	
	public int[][] filter(Instances data)
	{
		int datanum = data.numInstances();

	    int NumofAtt = data.numAttributes();

	    int[][] dataout = new int[datanum][NumofAtt];
	    for(int i = 0;i < datanum; i++) 
	    {
	    	int NumValues = data.instance(i).numValues();
	    	for(int j = 0; j < NumValues;j++) {
		    	int tmp = data.instance(i).index(j);
		    	dataout[i][tmp] = 1;
	    	}
	    }
	    return dataout;
	}
	
	protected boolean[] toBool(double[] data) {
		boolean[] bipartition = new boolean[data.length];
		for(int i=0;i<data.length;i++) {
			if(data[i] == 1.0)
				bipartition[i] = true;
			else
				bipartition[i] = false;
		}
		return bipartition;
	}
	

}
