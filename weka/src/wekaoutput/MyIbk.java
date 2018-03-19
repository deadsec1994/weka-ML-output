package wekaoutput;


	/*
	 *   This program is free software: you can redistribute it and/or modify
	 *   it under the terms of the GNU General Public License as published by
	 *   the Free Software Foundation, either version 3 of the License, or
	 *   (at your option) any later version.
	 *
	 *   This program is distributed in the hope that it will be useful,
	 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
	 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	 *   GNU General Public License for more details.
	 *
	 *   You should have received a copy of the GNU General Public License
	 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
	 */

	/*
	 *    IBk.java
	 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
	 *
	 */


	

	
	import weka.classifiers.UpdateableClassifier;
    import weka.classifiers.lazy.IBk;
	import weka.core.AdditionalMeasureProducer;
	import weka.core.Instance;
	import weka.core.Instances;
	import weka.core.OptionHandler;
	import weka.core.TechnicalInformationHandler;
	import weka.core.WeightedInstancesHandler;
	
	/**
	 <!-- globalinfo-start -->
	 * K-nearest neighbours classifier. Can select appropriate value of K based on cross-validation. Can also do distance weighting.<br/>
	 * <br/>
	 * For more information, see<br/>
	 * <br/>
	 * D. Aha, D. Kibler (1991). Instance-based learning algorithms. Machine Learning. 6:37-66.
	 * <p/>
	 <!-- globalinfo-end -->
	 * 
	 <!-- technical-bibtex-start -->
	 * BibTeX:
	 * <pre>
	 * &#64;article{Aha1991,
	 *    author = {D. Aha and D. Kibler},
	 *    journal = {Machine Learning},
	 *    pages = {37-66},
	 *    title = {Instance-based learning algorithms},
	 *    volume = {6},
	 *    year = {1991}
	 * }
	 * </pre>
	 * <p/>
	 <!-- technical-bibtex-end -->
	 *
	 <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -I
	 *  Weight neighbours by the inverse of their distance
	 *  (use when k &gt; 1)</pre>
	 * 
	 * <pre> -F
	 *  Weight neighbours by 1 - their distance
	 *  (use when k &gt; 1)</pre>
	 * 
	 * <pre> -K &lt;number of neighbors&gt;
	 *  Number of nearest neighbours (k) used in classification.
	 *  (Default = 1)</pre>
	 * 
	 * <pre> -E
	 *  Minimise mean squared error rather than mean absolute
	 *  error when using -X option with numeric prediction.</pre>
	 * 
	 * <pre> -W &lt;window size&gt;
	 *  Maximum number of training instances maintained.
	 *  Training instances are dropped FIFO. (Default = no window)</pre>
	 * 
	 * <pre> -X
	 *  Select the number of nearest neighbours between 1
	 *  and the k value specified using hold-one-out evaluation
	 *  on the training data (use when k &gt; 1)</pre>
	 * 
	 * <pre> -A
	 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
	 * </pre>
	 * 
	 <!-- options-end -->
	 *
	 * @author Stuart Inglis (singlis@cs.waikato.ac.nz)
	 * @author Len Trigg (trigg@cs.waikato.ac.nz)
	 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
	 * @version $Revision: 10141 $
	 */
	public class MyIbk 
	  extends IBk 
	  implements OptionHandler, UpdateableClassifier, WeightedInstancesHandler,
	             TechnicalInformationHandler, AdditionalMeasureProducer {

	  public MyIbk() {
			super();
			// TODO 自动生成的构造函数存根
		}

		public MyIbk(int k) {
			super(k);
			// TODO 自动生成的构造函数存根
		}

	@SuppressWarnings("unused")
	public double[]  distributionForInstance(Instance instance,int flag) throws Exception {

	    if (m_Train.numInstances() == 0) {
	      //throw new Exception("No training instances!");
	      return m_defaultModel.distributionForInstance(instance);
	    }
	    if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
	      m_kNNValid = false;
	      boolean deletedInstance=false;
	      while (m_Train.numInstances() > m_WindowSize) {
		m_Train.delete(0);
	      }
	      //rebuild datastructure KDTree currently can't delete
	      if(deletedInstance==true)
	        m_NNSearch.setInstances(m_Train);
	    }

	    // Select k by cross validation
	    if (!m_kNNValid && (m_CrossValidate) && (m_kNNUpper >= 1)) {
	      crossValidate();
	    }

	    m_NNSearch.addInstanceInfo(instance);
	   
	    Instances neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
//	    System.out.println(neighbours.numInstances());
	    double[] neighboursindices = new double[neighbours.numInstances()];
	    int index = 0;

	    for(int i=0;i<neighbours.numInstances();i++){
	    	Instance a = neighbours.get(i);
	    	for(int k=0;k<m_Train.numInstances();k++){
	    		Instance b = m_Train.get(k);
	    		if(myequals(a,b)&&judge(k + (k<flag? 1:2),neighboursindices)){
	    			if(k<flag)
	    				neighboursindices[index++] = (double)(k+1);
	    			else
	    				neighboursindices[index++] = (double)(k+2);
		    			break;
	    			}		
	    		}	
	    	}
	    
	    double [] distances = m_NNSearch.getDistances();
	    double [] distribution = makeDistribution( neighbours, distances );

		return neighboursindices;
	  }
	
	public boolean myequals(Instance a,Instance b){
		int temp=a.numAttributes();
//		for(int i = 0; i< temp; i++){
//			boolean it = a.value(i) != a.value(i) || !(a.value(i) == Double.NaN && a.value(i) == Double.NaN);
//			System.out.print(it + ",");
//		}
//		System.out.println();
		
		for(int i = 0; i < temp; i++){
			Double at = a.value(i);
			Double bt = b.value(i);
			if(!at.equals(bt)){
				if(at.isNaN() && bt.isNaN())
					continue;
				return false;
			}
//			else if(!(a.value(i) == Double.NaN && b.value(i) == Double.NaN))
//				return false;
		}
		return true;
	}
	public boolean judge(int a,double[] b){
		boolean temp=true;
		for(double i:b){
			if(((int)i)==a){
				temp=false;
				break;
			}
		}
		return temp;
	}
  	}

