package wekaoutput;

import java.io.FileWriter;

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
 *    LinearNNSearch.java
 *    Copyright (C) 1999-2012 University of Waikato
 */

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 <!-- globalinfo-start -->
 * Class implementing the brute force search algorithm for nearest neighbour search.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -S
 *  Skip identical instances (distances equal to zero).
 * </pre>
 * 
 <!-- options-end -->
 *
 * @author Ashraf M. Kibriya (amk14[at-the-rate]cs[dot]waikato[dot]ac[dot]nz)
 * @version $Revision: 10141 $
 */
public class MyLinearSearch
  extends LinearNNSearch {


  /**
   * Returns k nearest instances in the current neighbourhood to the supplied
   * instance.
   *  
   * @param target 	The instance to find the k nearest neighbours for.
   * @param kNN		The number of nearest neighbours to find.
   * @return		the k nearest neighbors
   * @throws Exception  if the neighbours could not be found.
   */
  public Instances kNearestNeighbours(Instance target, int kNN) throws Exception {
  
    //debug
    boolean print=false;

    if(m_Stats!=null)
      m_Stats.searchStart();
 
    MyHeap heap = new MyHeap(kNN);
    double distance; int firstkNN=0;
    for(int i=0; i<m_Instances.numInstances(); i++) {
      if(target == m_Instances.instance(i)) //for hold-one-out cross-validation
        continue;
      if(m_Stats!=null) 
        m_Stats.incrPointCount();
      if(firstkNN<kNN) {
        if(print)
          System.out.println("K(a): "+(heap.size()+heap.noOfKthNearest()));
        distance = m_DistanceFunction.distance(target, m_Instances.instance(i), Double.POSITIVE_INFINITY, m_Stats);
        if(distance == 0.0 && m_SkipIdentical)
          if(i<m_Instances.numInstances()-1)
            continue;
          else
            heap.put(i, distance);
        heap.put(i, distance);
        firstkNN++;
      }
      else {
        MyHeapElement temp = heap.peek();
        if(print)
          System.out.println("K(b): "+(heap.size()+heap.noOfKthNearest()));
        distance = m_DistanceFunction.distance(target, m_Instances.instance(i), temp.distance, m_Stats);
        if(distance == 0.0 && m_SkipIdentical)
          continue;
        if(distance < temp.distance) {
          heap.putBySubstitute(i, distance);
        }
        else if(distance == temp.distance) {
          heap.putKthNearest(i, distance);
        }

      }
    }
    
    Instances neighbours = new Instances(m_Instances, (heap.size()+heap.noOfKthNearest()));
    m_Distances = new double[heap.size()+heap.noOfKthNearest()];
    int [] indices = new int[heap.size()+heap.noOfKthNearest()];
    int i=1; MyHeapElement h;
    while(heap.noOfKthNearest()>0) {
      h = heap.getKthNearest();
      indices[indices.length-i] = h.index;
      m_Distances[indices.length-i] = h.distance;
      i++;
    }
    while(heap.size()>0) {
      h = heap.get();
      indices[indices.length-i] = h.index;
      m_Distances[indices.length-i] = h.distance;
      i++;
    }
    
    m_DistanceFunction.postProcessDistances(m_Distances);
    
    for(int k=0; k<indices.length; k++) {
      neighbours.add(m_Instances.instance(indices[k]));
    }
     
    if(m_Stats!=null)
      m_Stats.searchFinish();
    
    return neighbours;    
  }
//test123
}
