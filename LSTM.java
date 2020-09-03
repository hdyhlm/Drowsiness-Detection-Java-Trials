//FEED FROM TEST.CSV (FEATURES: EAR, MAR, CIR, MOUTH_EYE)

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class LSTM {
    private static int numSkipLines = 1;
    private static int batchSize = 10;
    private static double learningRate = 0.00005;
    private static int epoch =50;
    private static int tbpttLength = 5;

    public static void main(String[] args) throws IOException, InterruptedException {

//        File baseDir = new ClassPathResource("TEST.csv").getFile();
//        File featureDir = new File(baseDir, "feature");
//        File labelDir = new File(baseDir, "label");

        Schema inputSchema = new Schema.Builder()
                .addColumnString("EAR")
                .addColumnString("circularity")
                .addColumnString("MOE")
                .addColumnString("MAR")
                .addColumnString("Energy Level")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .convertToDouble("EAR")
                .convertToDouble("circularity")
                .convertToDouble("MOE")
                .convertToDouble("MAR")
                .build();

        File trainFileDir = new File(System.getProperty("user.home"), "Desktop/Group3/Group3/src/main/resources/train/");
        File trainFeaturesDir = new File(trainFileDir, "features");
        File trainLabelsDir = new File(trainFileDir, "labels");

        File testFileDir = new File(System.getProperty("user.home"), "Desktop/Group3/Group3/src/main/resources/test/");
        File testFeaturesDir = new File(testFileDir, "features");
        File testLabelsDir = new File(testFileDir, "labels");

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1,",");
        trainFeatures.initialize(new NumberedFileInputSplit(trainFeaturesDir.getAbsolutePath() + "/%d.csv", 0, 16));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(0,",");
        trainLabels.initialize(new NumberedFileInputSplit(trainLabelsDir.getAbsolutePath() + "/%d.csv", 0, 16));

        DataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize, 3, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(1,",");
        testFeatures.initialize(new NumberedFileInputSplit(testFeaturesDir.getAbsolutePath() + "/%d.csv", 0, 3));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader(0,",");
        testLabels.initialize(new NumberedFileInputSplit(testLabelsDir.getAbsolutePath() + "/%d.csv", 0, 3));

        DataSetIterator test = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, batchSize, 3, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        System.out.println("Hi");

//        File input = new File("Group3\\TEST.csv");
//
//        RecordReader rr = new CSVRecordReader(1, ',');
//        rr.initialize(new FileSplit(input));
//
//        List<List<Writable>> oriData = new ArrayList<>();
//        while (rr.hasNext()){
//            oriData.add(rr.next());
//        }
//
//        List<List<Writable>> processData = LocalTransformExecutor.execute(oriData, tp);
//
//        Collections.shuffle(processData, new Random((1234)));
//        int trainEnd = (int)(Math.ceil(processData.size() * 0.8));
//        List<List<Writable>> trainList = processData.subList(0, trainEnd);
//        List<List<Writable>> testList = processData.subList(trainEnd, processData.size());
//
//        DataSetIterator trainIter = new ListDataSetIterator(trainList, batchSize);
//        DataSetIterator testIter = new ListDataSetIterator(testList, batchSize);
//
//        RecordReader collect = new CollectionRecordReader(processData);
//        DataSetIterator iter = new RecordReaderDataSetIterator(collect, batchSize, 4, 3);
//
//        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iter, batchSize, 0.8);
//        List<DataSetIterator> trainAndTest = splitter.getIterators();
//        DataSetIterator train = trainAndTest.get(0);
//        DataSetIterator test = trainAndTest.get(1);
//
//        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numSkipLines,",");
//        trainFeatures.initialize(new NumberedFileInputSplit( featureDir.getAbsolutePath()+ "/%d.csv", 0, 3));
//        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(numSkipLines, ",");
//        trainLabels.initialize(new NumberedFileInputSplit(labelDir.getAbsolutePath()+"/%d.csv", 0, 3));
//
//        DataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize,1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
//
          //int numInput = train.inputColumns();


        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .dropOut(0.5)
                .list()
//                .layer(new DenseLayer.Builder()
//                        .nIn(4)
//                        .nOut(1024)
//                        .activation(Activation.SIGMOID)
//                        .build())
                .layer(new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                        .nIn(4)
                        .nOut(512)
                        .activation(Activation.TANH)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(216)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(32)
                        .activation(Activation.TANH)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(216)
                        .activation(Activation.TANH)
                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .nIn(216)
                        .nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTLength(tbpttLength)
                .build();

        System.out.println("hola");

        StatsStorage stor = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(stor);

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new StatsListener(stor,10));

        for (int i=0; i<epoch; i++) {
            System.out.println("Epoch: " + i);
            model.fit(train);
            train.reset();
        }

        File locationSave = new File("trained_model.zip");
        boolean saveupd = true;
        ModelSerializer.writeModel(model, locationSave, saveupd);
        System.out.println("\nTrain network saved at " + locationSave);

        System.out.println("***** Test Evaluation *****");
        Evaluation eval = new Evaluation(3);
        test.reset();
        DataSet testData = test.next(1);
        INDArray IA = testData.getFeatures();
        System.out.println(IA);

        while (test.hasNext())
        {
            testData = test.next();
            INDArray predicted = model.output(testData.getFeatures());
            INDArray labels = testData.getLabels();

            eval.evalTimeSeries(labels, predicted, testData.getLabelsMaskArray());
        }

        System.out.println(eval.stats());

    }
}


