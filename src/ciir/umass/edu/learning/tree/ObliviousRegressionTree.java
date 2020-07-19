package ciir.umass.edu.learning.tree;

import ciir.umass.edu.learning.DataPoint;

import java.util.ArrayList;
import java.util.List;

public class ObliviousRegressionTree extends RegressionTree {

    int treedepth = 10;
    int[] sampleids = new int[super.trainingSamples.length];
    float invalid = Float.MAX_VALUE;

    public ObliviousRegressionTree(Split root) {
        super(root);
    }

    public ObliviousRegressionTree(int nLeaves, DataPoint[] trainingSamples, double[] labels, FeatureHistogram hist, int minLeafSupport) {
        super(nLeaves, trainingSamples, labels, hist, minLeafSupport);
    }

    /**
     * Fit the tree from the specified training data
     */
    @Override
    public void fit() {
        //change according to oblivious tree
        for (int a = 0; a < trainingSamples.length; a++) {
            sampleids[a] = a;
        }

        int nFeatureSamples = hist.features.length;
        Split[] nodeArray = new Split[(1 << (treedepth + 1))];
        nodeArray[0] = root = new Split(sampleids, hist);
        List<float[]> sumScores = new ArrayList<>();

        for (int i = 0; i < nFeatureSamples; ++i) {
            sumScores.add(new float[hist.count[i].length]);
        }

        for (int depth = 0; depth < treedepth; ++depth) {
            int lbegin = (1 << depth) - 1;
            int lend = (1 << depth + 1) - 1;
            for (int i = 0; i < nFeatureSamples; ++i) {
                int thresholdsSize = hist.count[i].length;
                for (int j = 0; j < thresholdsSize; ++j) {
                    sumScores.get(i)[j] = new Float(.0f);
                }
            }
            //for each histogram on the current depth (i.e. fringe) add variance of each (feature,threshold) in sumvar matrix
            for (int i = lbegin; i < lend; ++i) {
                fill(sumScores, nFeatureSamples, nodeArray[i].hist);
            }
            //find best split in the matrix
            double maxScore = 0.0;
            int bestFeatureIdx = Integer.MAX_VALUE;
            int bestThresholdId = Integer.MAX_VALUE;

            for (int f = 0; f < nFeatureSamples; ++f) {
                int thresholdSize = hist.count[f].length;
                for (int t = 0; t < thresholdSize; ++t) {
                    if (sumScores.get(f)[t] != invalid && sumScores.get(f)[t] > maxScore) {
                        maxScore = sumScores.get(f)[t];
                        bestFeatureIdx = f;
                        bestThresholdId = t;
                    }
                }
            }

            if (maxScore == invalid || maxScore == 0.0) {
                break; //node is unsplittable
            }
            //init next depth
            for (int i = lbegin; i < lend; ++i) {
                Split split = nodeArray[i];
                int[] sampleIds = new int[split.getSamples().length];
                for (int j = 0; j < sampleIds.length; j++) {
                    sampleIds[j] = j;
                }

                int lastThresholdId = split.hist.count[bestFeatureIdx].length - 1;
                int leftCount = split.hist.count[bestFeatureIdx][bestThresholdId];
                int rightCount = split.hist.count[bestFeatureIdx][lastThresholdId] - leftCount;
                float bestThreshold = split.hist.thresholds[bestFeatureIdx][bestThresholdId];

                int[] leftSamples = new int[leftCount];
                int lsize = 0;
                int[] rightSamples = new int[rightCount];
                int rsize = 0;

                float[] features = new float[trainingSamples.length];
                for (int j = 0; j < trainingSamples.length; j++) {
                    features[j] = trainingSamples[j].getfVals()[bestFeatureIdx+1];
                }

                for (int j = 0, nsampleids = sampleIds.length; j < nsampleids; ++j) {
                    int k = split.getSamples()[j];
                    if (features[k] <= bestThreshold) {
                        leftSamples[lsize++] = k;
                    } else {
                        rightSamples[rsize++] = k;
                    }
                }
                FeatureHistogram leftHist;
                FeatureHistogram rightHist;
                if (depth != treedepth - 1) {
                    leftHist = new FeatureHistogram(split.hist, leftSamples, lsize, trainingLabels);
                    if (split.equals(root)) {
                        rightHist = new FeatureHistogram(split.hist, leftHist);
                    } else {
                        split.hist.transformIntoRightChild(leftHist);
                        rightHist = split.hist;
                        split.hist = null;
                    }
                    nodeArray[2 * i + 1] = new Split(leftSamples, leftHist);
                    nodeArray[2 * i + 2] = new Split(rightSamples, rightHist);
                    //update current node
                    split.setLeft(nodeArray[2 * i + 1]);
                    split.setRight(nodeArray[2 * i + 2]);
                } else {
                    double lsum = split.hist.sum[bestFeatureIdx][bestThresholdId];
                    double rsum = split.hist.sum[bestFeatureIdx][lastThresholdId] - lsum;
                    nodeArray[2 * i + 1] = new Split(leftSamples, lsum / lsize);
                    nodeArray[2 * i + 2] = new Split(rightSamples, rsum / rsize);
                    split.setLeft(nodeArray[2 * i + 1]);
                    split.setRight(nodeArray[2 * i + 2]);
                }
                split.setFeatureID(bestFeatureIdx + 1);
                split.setThreshold(bestThreshold);
            }
        }
        leaves = root.leaves();
    }

    private void fill(List<float[]> sumVar, int nFeatureSamples, FeatureHistogram hist) {
        for (int f = 0; f < nFeatureSamples; ++f) {
            double[] sumLabel = hist.sum[f];
            int[] sampleCount = hist.count[f];
            int thresholdSize = hist.thresholds[f].length;

            double s = sumLabel[thresholdSize - 1];
            int c = sampleCount[thresholdSize - 1];

            for (int t = 0; t < thresholdSize; ++t)
                if (sumVar.get(f)[t] != invalid) {
                    int countLeft = sampleCount[t];
                    int countRight = c - countLeft;
                    if (countLeft >= minLeafSupport && countRight >= minLeafSupport) {
                        double sumLeft = sumLabel[t];
                        double sumRight = s - sumLeft;
                        sumVar.get(f)[t] += sumLeft * sumLeft / countLeft + sumRight * sumRight / countRight;
                    } else
                        sumVar.get(f)[t] = invalid;
                }
        }
    }
}
