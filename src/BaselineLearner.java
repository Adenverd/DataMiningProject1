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

    /**
     * Performs n-fold cross-validation on the data that was already trained on
     * @param n
     * @return
     */
    public static double nFoldCrossValidate(Matrix features, Matrix labels, int n){
        int featureRows = features.getNumRows();

        BaselineLearner learner = new BaselineLearner();

        if(n > featureRows){
            throw new MLException(String.format("Cannot cross-validate with %d folds on a %d row matrix", n, featureRows));
        }

        int foldSize;
        //Determine foldSize
        if (featureRows%n == 0){
            foldSize = featureRows/n;
        }
        else{
            foldSize = (featureRows/n) + 1;
        }

        double sum = 0.0;
        //for each fold
        for (int fold = 0; fold < n; fold++){
            //Determine fold start and end rows
            int foldStartRow = fold*foldSize;
            int foldEndRow = (fold+1)*foldSize > featureRows? featureRows : (fold+1)*foldSize;

            Matrix tempFeatures = new Matrix(features);
            Matrix tempLabels = new Matrix(labels);

            Matrix foldFeatures = tempFeatures.removeFold(foldStartRow, foldEndRow);
            Matrix foldLabels = tempLabels.removeFold(foldStartRow, foldEndRow);

            learner.train(tempFeatures, tempLabels);

            sum += learner.getAccuracy(foldFeatures, foldLabels);
        }

        return sum;
    }
}
