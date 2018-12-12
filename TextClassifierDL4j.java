import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;


public class TextClassifierDL4j {
    private static String train_vec_source = "data_keyword"+File.separator+"train4vec.txt";
    private static String train_source  = "data_keyword"+File.separator+"train.txt";
    private static String test_source  = "data_keyword"+File.separator+"test.txt";
//    private static String w2v_model = "model"+File.separator+"w2v.bin";
    private static String w2v_model = "model"+File.separator+"fasttext.vec";
    private static String cnn_model = "model"+File.separator+"cnn.bin";
    private static String stopwords_file = "stopwords.txt";

    private static final Logger log = LoggerFactory.getLogger(KeywordClassifierDL4j.class);
    private static int batchSize = 64;
    private static long seed = 42;
    private static Random rng = new Random(seed);
    private static int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

    private static TokenizerFactory tokenizerfactory(){
        TokenizerFactory tf = new DefaultTokenizerFactory();
//        TokenizerFactory tfn = new NGramTokenizerFactory(tf,1,3);
        tf.setTokenPreProcessor(new CommonPreprocessor());
        return tf;
    }

    private static void word2vec_train() throws Exception{
        SentenceIterator iter = new BasicLineIterator(train_vec_source);
        TokenizerFactory tf = tokenizerfactory();
        List<String> stopwords = PrepareData.readTxtFileIntoStringArrList(stopwords_file);
        System.out.println("building word2vec will start");
        Word2Vec w2v = new Word2Vec.Builder()
                .minWordFrequency(0)
                .iterations(10)
                .batchSize(batchSize)
                .layerSize(300)
                .seed(seed)
//                .stopWords(stopwords)
                .windowSize(10)
                .iterate(iter)
                .tokenizerFactory(tf)
                .build();
        w2v.fit();
        WordVectorSerializer.writeWordVectors(w2v, w2v_model);
        System.out.println("building word2vec fin");
    }

    private static void word2vec_test() throws Exception{
        Word2Vec w2v = WordVectorSerializer.readWord2VecModel(new File(w2v_model)); //可以用gensim训练的模型
        System.out.println("xiaomi Closest Words:");
        Collection<String> lst = w2v.wordsNearest(PrepareData.preProcess_keyword("xiaomi"), 5);
        System.out.println(lst);
    }

    private static DataSetIterator getDataSetIterator(boolean isTraining, Word2Vec wordVectors,int sentenceLength, int miniBatch, Random rng){
        List<String> labels = new ArrayList<>();
        List<String> sentences = new ArrayList<>();
        List<String> title_list  = PrepareData.readTxtFileIntoStringArrList(isTraining ? train_source : test_source);
        String[] lt;
        for(String title: title_list) {
            lt = title.split("\\t+",2);
            labels.add(lt[0]);
            sentences.add(lt[1]);
        }
        LabeledSentenceProvider p = new CollectionLabeledSentenceProvider(sentences, labels, rng);

        return new CnnSentenceDataSetIterator.Builder()
                .sentenceProvider(p)
                .wordVectors(wordVectors)
                .maxSentenceLength(sentenceLength)
                .minibatchSize(miniBatch)
                .useNormalizedWordVectors(true)
                .build();
    }


    private ComputationGraphConfiguration kimconfig(int vectorSize, int cnnLayerFeatureMaps){
        return new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.1))
                .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
                .l2(0.001)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn2", new ConvolutionLayer.Builder()
                        .kernelSize(2,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn3", new ConvolutionLayer.Builder()
                        .kernelSize(3,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn4", new ConvolutionLayer.Builder()
                        .kernelSize(4,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addLayer("cnn5", new ConvolutionLayer.Builder()
                        .kernelSize(5,vectorSize)
                        .stride(1,vectorSize)
                        .nIn(1)
                        .nOut(cnnLayerFeatureMaps)
                        .build(), "input")
                .addVertex("merge", new MergeVertex(), "cnn2", "cnn3", "cnn4","cnn5")      //Perform depth concatenation
                .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .dropOut(0.5)
                        .build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(4*cnnLayerFeatureMaps)
                        .nOut(11)
                        .build(), "globalPool")
                .setOutputs("out")
                .build();
    }



    private void cnn() throws Exception{
        Word2Vec w2v = WordVectorSerializer.readWord2VecModel(new File(w2v_model));
        TokenizerFactory tf = tokenizerfactory();
        w2v.setTokenizerFactory(tf);

        //Basic configuration
        int vectorSize = w2v.lookupTable().layerSize();               //Size of the word vectors.
        int nEpochs = 1;                    //Number of epochs (full passes of training data) to train on
        int cnnLayerFeatureMaps = 100;      //Number of feature maps / channels / depth for each CNN layer
        Random rng = new Random(12345); //For shuffling repeatability

        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        ComputationGraph net = new ComputationGraph(kimconfig(vectorSize, cnnLayerFeatureMaps));
//        ComputationGraph net = new ComputationGraph(consumeconfig(vectorSize, cnnLayerFeatureMaps));
        net.init();

        System.out.println("Number of parameters by layer:");
        for(Layer l : net.getLayers() ){
            System.out.println("\t" + l.conf().getLayer().getLayerName() + "\t" + l.numParams());
        }

        System.out.println("Loading word vectors and creating DataSetIterators");
        DataSetIterator trainIter = getDataSetIterator(true, w2v, truncateReviewsToLength, batchSize, rng);
        DataSetIterator testIter  = getDataSetIterator(false, w2v, truncateReviewsToLength, batchSize, rng);
        System.out.println("Starting training");

        net.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Epoch " + i + " complete. Starting evaluation:");
            //Run evaluation. This is on 25k reviews, so can take some time
//            Evaluation evaluation = net.evaluate(testIter);
//            System.out.println(evaluation.stats());
        }
        ModelSerializer.writeModel(net,cnn_model,true);
    }

    private void cnn_test() throws Exception{
        Word2Vec w2v = WordVectorSerializer.readWord2VecModel(new File(w2v_model));
        TokenizerFactory tf = tokenizerfactory();
        w2v.setTokenizerFactory(tf);
        ComputationGraph net = ModelSerializer.restoreComputationGraph(cnn_model);
        DataSetIterator trainIter = getDataSetIterator(true, w2v, truncateReviewsToLength, batchSize, rng);
        DataSetIterator testIter = getDataSetIterator(false, w2v, truncateReviewsToLength, batchSize, rng);
//        Evaluation evaluation = net.evaluate(testIter);
//        System.out.println(evaluation.stats());
        List<String> title_list  = PrepareData.readTxtFileIntoStringArrList(test_source);
        List<String> labels = trainIter.getLabels();
        INDArray features,predictions;
        String[] lt;
        for(String title: title_list) {
            lt = title.split("\\t+",2);
            try {
                features = ((CnnSentenceDataSetIterator) trainIter).loadSingleSentence(lt[1]);
                predictions = net.outputSingle(features);
            }catch (Exception e){
                continue;
            }
            System.out.println("\n\n"+lt[1]+": ");
            for (int i = 0; i < labels.size(); i++) {
                System.out.println("P(" + labels.get(i) + ") = " + predictions.getDouble(i));
            }
        }
    }

    public static void main(String[] args) throws  Exception{
//        word2vec_train();
//        word2vec_test();
        TextClassifierDL4j cls = new TextClassifierDL4j();
        cls.cnn();
        cls.cnn_test();

    }
}
