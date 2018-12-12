import weka.associations.Apriori;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomialText;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.*;

import weka.filters.supervised.instance.ClassBalancer;
import weka.associations.FilteredAssociator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;

import java.io.*;

public class Weka {

    private static void csv2arff(String path) throws Exception{
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("weka"+File.separator+path+".csv"));
        Instances data = loader.getDataSet();

        MultiFilter mf = new MultiFilter();
        Filter[] fs = new Filter[2];
        NominalToString nts = new NominalToString();
        nts.setAttributeIndexes("first");
        NumericToNominal ntn = new NumericToNominal();
        ntn.setAttributeIndices("last");
        fs[1] = nts;
        fs[0] = ntn;

        mf.setInputFormat(data);
        mf.setFilters(fs);

        for (int i = 0; i < data.numInstances(); i++) {
            mf.input(data.instance(i));
        }
        mf.batchFinished();

        Instances newData = mf.getOutputFormat();
        Instance processed;
        while ((processed = mf.output()) != null) {
            newData.add(processed);
        }

        ArffSaver saver_test = new ArffSaver();
        System.out.println(path+" save...");
        saver_test.setInstances(newData);
        saver_test.setFile(new File("weka"+File.separator+path+".arff"));
        saver_test.writeBatch();
        
    }

    private static void tfidf() throws Exception{
        DataSource train_source = new DataSource("weka"+File.separator+"train.arff");
        Instances train_data = train_source.getDataSet();
        DataSource test_source = new DataSource("weka"+File.separator+"test.arff");
        Instances test_data = test_source.getDataSet();

        StringToWordVector filter = new StringToWordVector();
        filter.setAttributeIndices("first");
        filter.setInputFormat(train_data);
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        filter.setMinTermFreq(3);
        filter.setWordsToKeep(1000000);
//        filter.setAttributeNamePrefix("wv_");
        NGramTokenizer ngt = new NGramTokenizer();
        ngt.setNGramMinSize(1);
        ngt.setNGramMaxSize(3);
        filter.setTokenizer(ngt);
        filter.setDictionaryFileToSaveTo(new File("weka"+File.separator+"train.dict"));
        filter.setOutputWordCounts(true);

        System.out.println("tfidf train...");
        Instances newTrain = Filter.useFilter(train_data,filter);

        System.out.println("tfidf test...");
        Instances newTest = Filter.useFilter(test_data,filter);

        //save
        ArffSaver saver_train = new ArffSaver();
        System.out.println("train save...");
        saver_train.setInstances(newTrain);
        saver_train.setFile(new File("weka"+File.separator+"train_vec.arff"));
        saver_train.writeBatch();

        ArffSaver saver_test = new ArffSaver();
        System.out.println("test save...");
        saver_test.setInstances(newTest);
        saver_test.setFile(new File("weka"+File.separator+"test_vec.arff"));
        saver_test.writeBatch();
    }

    private static void discretize() throws Exception{
        DataSource source = new DataSource("weka"+File.separator+"train_vec.arff");
        Instances data = source.getDataSet();
        System.out.println("discriteze...");
        Discretize d = new Discretize();
        d.setInputFormat(data);
        Instances newData= Filter.useFilter(data, d);
        System.out.println("discriteze save...");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File("weka"+File.separator+"train_vec_d.arff"));
        saver.writeBatch();
    }

    private static void associatorDiscretize() throws Exception{
        DataSource source = new DataSource("weka"+File.separator+"train_vec.arff");
        Instances data = source.getDataSet();
        FilteredAssociator fa = new FilteredAssociator();
        Discretize d = new Discretize();
        d.setInputFormat(data);
        Apriori a = new Apriori();
        fa.setFilter(d);
        fa.setAssociator(a);
        fa.buildAssociations(data);
    }

    private static void selectAttributs() throws Exception{
        DataSource source = new DataSource("weka"+File.separator+"train_vec.arff");
        Instances data = source.getDataSet();
        InfoGainAttributeEval igae = new InfoGainAttributeEval();
        Ranker r = new Ranker();
        igae.buildEvaluator(data);
    }

    private static void incrementalClassifier() throws Exception{
        System.out.println("data loading....");
        // load data
        DataSource train_source = new DataSource("weka"+File.separator+"train_vec.arff");
        Instances train_data = train_source.getDataSet();

        DataSource test_source = new DataSource("weka"+File.separator+"test_vec.arff");
        Instances test_data = test_source.getDataSet();

        //set Attribute
        train_data.setClassIndex(train_data.numAttributes() - 1);
        test_data.setClassIndex(test_data.numAttributes()-1);
        System.out.println("classify init....");
        // meta-classifier
//        ZeroR cls = new ZeroR();
        SMOreg cls = new SMOreg();
//        J48 cls = new J48();
        // train and make predictions

        Evaluation eval = new Evaluation(train_data);
        eval.setDiscardPredictions(true);

        System.out.println("filter init....");
        ClassBalancer cb = new ClassBalancer();
        cb.setInputFormat(train_data);
//
        System.out.println("FilteredClassifier....");
        FilteredClassifier fc = new FilteredClassifier();
        fc.setClassifier(cls);
        fc.setFilter(cb);
        fc.buildClassifier(train_data);
//
//        //evaluation
        System.out.println("Evaluation....");
        eval.evaluateModel(fc, test_data);

//        cls.buildClassifier(train_data);
//        eval.evaluateModel(cls, test_data);

        System.out.println(eval.toSummaryString("\nResults\n======\n", true));
        System.out.println(eval.toMatrixString("\"\\nMatrix Confusion\\n======\\n\""));
        System.out.println(eval.toClassDetailsString("\nClass Details\n======\n"));
        System.out.println(eval.incorrect());

    }

    private static void nbtextcls() throws Exception{
        System.out.println("data loading....");
        // load data
        DataSource train_source = new DataSource("weka"+File.separator+"train.arff");
        Instances train_data = train_source.getDataSet();

        DataSource test_source = new DataSource("weka"+File.separator+"test.arff");
        Instances test_data = test_source.getDataSet();

        //set Attribute
        train_data.setClassIndex(train_data.numAttributes() - 1);
        test_data.setClassIndex(test_data.numAttributes()-1);
        System.out.println("classify init....");
        NaiveBayesMultinomialText cls = new NaiveBayesMultinomialText();

        NGramTokenizer ngt = new NGramTokenizer();
        ngt.setNGramMinSize(1);
        ngt.setNGramMaxSize(3);
        cls.setTokenizer(ngt);
        cls.setMinWordFrequency(1.0);
        WordsFromFile stopword = new WordsFromFile();
        stopword.setStopwords(new File("stopwords.txt"));
        cls.setStopwordsHandler(stopword);
        Evaluation eval= null;
        System.out.println("Trainning....");
        eval = new Evaluation(train_data);
        cls.buildClassifier(train_data);
        eval.evaluateModel(cls, test_data);

        System.out.println(eval.toSummaryString("\nResults\n======\n", true));
        System.out.println(eval.toMatrixString("\"\nMatrix Confusion\n======\n\""));
        System.out.println(eval.toClassDetailsString("\nClass Details\n======\n"));
        System.out.println(eval.incorrect());
        SerializationHelper.write("weka"+File.separator+"nbmt.model",cls);
    }

    public static void main(String[] args) throws  Exception{
//        csv2arff("test");
//        csv2arff("train");
//        tfidf();
//        discretize();
        incrementalClassifier();
//        nbtextcls();
    }

}
