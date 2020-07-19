package ciir.umass.edu.learning.tree;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.utilities.SimpleMath;

import java.util.List;

public class ObliviousLambdaMART extends LambdaMART {
    static { nTreeLeaves = 3; }

    public ObliviousLambdaMART() {
    }

    public void learn() {
        ensemble = new Ensemble();

        PRINTLN("---------------------------------");
        PRINTLN("Training starts...");
        PRINTLN("---------------------------------");
        PRINTLN(new int[]{7, 9, 9}, new String[]{"#iter", scorer.name() + "-T", scorer.name() + "-V"});
        PRINTLN("---------------------------------");

        //Start the gradient boosting process
        for (int m = 0; m < nTrees; m++) {
            PRINT(new int[]{7}, new String[]{(m + 1) + ""});

            //Compute lambdas (which act as the "pseudo responses")
            //Create training instances for MART:
            //  - Each document is a training sample
            //	- The lambda for this document serves as its training label
            computePseudoResponses();

            //update the histogram with these training labels (the feature histogram will be used to find the best tree split)
            int[] sampleids = new int[martSamples.length];
            for (int a = 0; a < sampleids.length; a++) {
                sampleids[a] = a;
            }
            hist.update(pseudoResponses, sampleids);

            //Fit a regression tree
            ObliviousRegressionTree rt = new ObliviousRegressionTree(nTreeLeaves, martSamples, pseudoResponses, hist, minLeafSupport);
            rt.fit();

            //Add this tree to the ensemble (our model)
            ensemble.add(rt, learningRate);

            //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
            updateTreeOutput(rt);

            //Update the model's outputs on all training samples
            List<Split> leaves = rt.leaves();
            for (int i = 0; i < leaves.size(); i++) {
                Split s = leaves.get(i);
                int[] idx = s.getSamples();
                for (int j = 0; j < idx.length; j++)
                    modelScores[idx[j]] += learningRate * s.getOutput();
            }

            //clear references to data that is no longer used
            rt.clearSamples();

            //beg the garbage collector to work...
            if (m % gcCycle == 0)
                System.gc();//this call is expensive. We shouldn't do it too often.

            //Evaluate the current model
            scoreOnTrainingData = computeModelScoreOnTraining();
            //**** NOTE ****
            //The above function to evaluate the current model on the training data is equivalent to a single call:
            //
            //		scoreOnTrainingData = scorer.score(rank(samples);
            //
            //However, this function is more efficient since it uses the cached outputs of the model (as opposed to re-evaluating the model
            //on the entire training set).

            PRINT(new int[]{9}, new String[]{SimpleMath.round(scoreOnTrainingData, 4) + ""});

            //Evaluate the current model on the validation data (if available)
            if (validationSamples != null) {
                //Update the model's scores on all validation samples
                for (int i = 0; i < modelScoresOnValidation.length; i++)
                    for (int j = 0; j < modelScoresOnValidation[i].length; j++)
                        modelScoresOnValidation[i][j] += learningRate * rt.eval(validationSamples.get(i).get(j));

                //again, equivalent to scoreOnValidation=scorer.score(rank(validationSamples)), but more efficient since we use the cached models' outputs
                double score = computeModelScoreOnValidation();

                PRINT(new int[]{9}, new String[]{SimpleMath.round(score, 4) + ""});
                if (score > bestScoreOnValidationData) {
                    bestScoreOnValidationData = score;
                    bestModelOnValidation = ensemble.treeCount() - 1;
                }
            }

            PRINTLN("");

            //Should we stop early?
            if (m - bestModelOnValidation > nRoundToStopEarly)
                break;
        }

        //Rollback to the best model observed on the validation data
        while (ensemble.treeCount() > bestModelOnValidation + 1)
            ensemble.remove(ensemble.treeCount() - 1);

        //Finishing up
        scoreOnTrainingData = scorer.score(rank(samples));
        PRINTLN("---------------------------------");
        PRINTLN("Finished sucessfully.");
        PRINTLN(scorer.name() + " on training data: " + SimpleMath.round(scoreOnTrainingData, 4));
        if (validationSamples != null) {
            bestScoreOnValidationData = scorer.score(rank(validationSamples));
            PRINTLN(scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
        }
        PRINTLN("---------------------------------");
    }

    public double eval(DataPoint dp) {
        return ensemble.eval(dp);
    }

    @Override
    public Ranker createNew() {
        return new ObliviousLambdaMART();
    }

    @Override
    public String name() {
        return "Oblivious LambdaMART";
    }

}
