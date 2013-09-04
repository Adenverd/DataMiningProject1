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
    public List<Double> nFoldCrossValidate(int n){
        if(n > featuresSize){
            throw new MLException(String.format("Cannot cross-validate with %d folds on a %d row matrix", n, featuresSize));
        }
        if (features == null){
            throw new MLException("Please train once before you cross-validate");
        }

        int foldSize;
        //Determine foldSize
        if (featuresSize%n == 0){
            foldSize = featuresSize/n;
        }
        else{
            foldSize = (featuresSize/n) + 1;
        }

        //for each fold
        for (int currentFold = 0; currentFold < n; currentFold++){
            int foldStart = currentFold*foldSize;
            int foldEnd = currentFold*(foldSize+1);
            Matrix foldFeatures;
            Matrix foldLabels;

            //If we're on the last fold, set aside the remaining rows
            if(currentFold == n-1){
                foldFeatures = features.subMatrix(currentFold*foldSize, n);
                foldLabels = features.subMatrix(currentFold*foldSize, n);
            }

            //Otherwise, set aside foldSize rows
            else {
                foldFeatures = features.subMatrix(currentFold*foldSize, (currentFold+1)*n);
                foldLabels = features.subMatrix(currentFold*n, (currentFold+1)*n);
            }

            for(int j = 0; j < featuresSize; j++){
                if(//j is not in the current fold)
            }
        }

        return null;
    }
}
