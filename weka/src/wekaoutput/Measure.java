package wekaoutput;


public class Measure {
	protected double sum;
    /**
     * The number of validation examples processed
     */
    protected int count;

	public void reset() {
	        sum = 0;
	        count = 0;
	    }


	public double getValue() {
	    return sum / count;
	    }
	
	protected void Accuracy(boolean[] bipartition, boolean[] truth) {
        double intersection = 0;
        double union = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i] && truth[i]) {
                intersection++;
            }
            if (bipartition[i] || truth[i]) {
                union++;
            }
        }

        if (union == 0) {
            sum += 1;
        } else {
            sum += intersection / union;
        }
        count++;
    }
	protected void Precision(boolean[] bipartition, boolean[] truth) {
        double tp = 0;
        double fp = 0;
        double fn = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i]) {
                if (truth[i]) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (truth[i]) {
                    fn++;
                }
            }
        }
        if (tp + fp + fn == 0) {
        	sum +=  1;
        }
        if (tp + fp == 0) {
        	sum +=  0;
        }
        sum +=  tp / (tp + fp);
        count++;
    }
    
    protected void Recall(boolean[] bipartition, boolean[] truth) {
        double tp = 0;
        double fp = 0;
        double fn = 0;
        for (int i = 0; i < truth.length; i++) {
            if (bipartition[i]) {
                if (truth[i]) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                if (truth[i]) {
                    fn++;
                }
            }
        }
        if (tp + fp + fn == 0) {
        	sum += 1;
        }
        if (tp + fn == 0) {
        	sum += 0;
        }
        sum += tp / (tp + fn);
        count++;
    }

}
