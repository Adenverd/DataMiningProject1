import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;
import static java.lang.Math.sqrt;

import java.util.*;

public class BaselineLearner extends SupervisedLearner {

    List<Double> columnMeans;
    Matrix features, labels;
    int featuresSize;
    int labelsSize;

    public BaselineLearner(){
        columnMeans = new ArrayList<Double>();
    }

    @Override
    /**
     * Since this is the BaselineLearner, we don't care about the features at all,
     * we only care about the labels in the training data. That is, the feature columns
     * have no effect on what the label column values is columnMeans will be.
     */
    public void train(Matrix features, Matrix labels) {
        this.features = features;
        this.labels = labels;
        featuresSize = features.getCols();
        labelsSize = labels.getCols();

        for(int col = 0; col < labels.getCols(); col++){
            columnMeans.add(labels.columnMean(col));
        }
    }

    @Override
    //Predict is assuming the the values are numerical/continuous
    public void predict(List<Double> in, List<Double> out) {
        out.clear();

        if(in.size()!= featuresSize){
            throw new MLException("Feature sizes don't match");
        }

        for(Double mean : columnMeans){
            out.add(mean);
        }
    }

    public double getAccuracy() {
        double sum = 0;
        for (int i = 0; i < features.getRows(); i++) {
            List<Double> out = new ArrayList<Double>();
            List<Double> in = features.row(i);

            predict(in, out);

            double magnitude = 0;
            for (int j = 0; j < labels.getCols(); j++) {
                double result = labels.row(i).get(j) - out.get(j);
                magnitude += result * result;
            }
            sum += magnitude;
        }
        return sum;
    }
}
