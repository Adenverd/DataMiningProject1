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
        featuresSize = features.getNumCols();
        labelsSize = labels.getNumCols();

        for(int col = 0; col < labels.getNumCols(); col++){
            columnMeans.add(labels.columnMean(col));
        }
    }

    @Override
    public List<Double> predict(List<Double> in) {
        if (in.size()!=featuresSize){
            throw new MLException(String.format("Unable to predict: in has %d rows, but learner was trained on %d rows.", in.size(), featuresSize));
        }
        return columnMeans;
    }

    public List<Double> nFoldCrossValidate(int n){
        if(n > featuresSize){
            throw new MLException(String.format("Cannot cross-validate with %d folds on a %d row matrix", n, featuresSize));
        }
        if (features == null){
            throw new MLException("Please train once before you cross-validate");
        }

        List<Double>

    }
}
