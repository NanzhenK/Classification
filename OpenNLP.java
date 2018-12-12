import opennlp.tools.cmdline.doccat.DoccatEvaluationErrorListener;
import opennlp.tools.cmdline.doccat.DoccatFineGrainedReportListener;
import opennlp.tools.doccat.*;
import opennlp.tools.util.*;
import opennlp.tools.util.eval.EvaluationMonitor;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class KeywordClassifierOpenNLP {

    private static void train(){
        try {
            InputStreamFactory dataIn = new InputStreamFactory() {
                public InputStream createInputStream() throws IOException {
                    return new FileInputStream("data"+File.separator+"train.txt");
                }
            };
            ObjectStream<String> lineStream =
                    new PlainTextByLineStream(dataIn, "UTF-8");
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);
            TrainingParameters params = new TrainingParameters();

//            params.put(TrainingParameters.ALGORITHM_PARAM, PerceptronTrainer.PERCEPTRON_VALUE);
//            params.put(TrainingParameters.ALGORITHM_PARAM, NaiveBayesTrainer.NAIVE_BAYES_VALUE);
//            params.put(TrainingParameters.TRAINER_TYPE_PARAM, AbstractEventTrainer.EVENT_VALUE);
//            params.put(TrainingParameters.TRAINER_TYPE_PARAM, EventModelSequenceTrainer.SEQUENCE_VALUE);

            params.put(TrainingParameters.CUTOFF_PARAM, 0);
            params.put("DataIndexer", "TwoPass");
            params.put(TrainingParameters.ITERATIONS_PARAM, 10);
            params.put(TrainingParameters.THREADS_PARAM, 10);

            FeatureGenerator[] featureGenerators = {
                    new BagOfWordsFeatureGenerator(),
                    new NGramFeatureGenerator(1,4),
            };

            DoccatFactory factory = new DoccatFactory(featureGenerators);

            DoccatModel model = DocumentCategorizerME.train("en", sampleStream, params, factory);
            System.out.println("\nModel is successfully trained.");

            BufferedOutputStream modelOut = new BufferedOutputStream(new FileOutputStream("model"+File.separator+"model_name.bin"));
            model.serialize(modelOut);
            System.out.println("\nTrained Model is saved locally at : "+"model"+File.separator+"model_name.bin");

        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, Double> sort_map(Map<String, Double> map){
        List<Map.Entry<String,Double>> list = new ArrayList<Map.Entry<String,Double>>(map.entrySet());
        list.sort(new Comparator<Map.Entry<String,Double>>() {
            public int compare(Map.Entry<String, Double> o1,
                               Map.Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }

        });
        map = new HashMap<>();
        for(Map.Entry<String,Double> mapping:list){
            map.put(mapping.getKey(), mapping.getValue());
        }
        return map;
    }
    
    private static void test() throws Exception{
        InputStream is = new FileInputStream("model"+File.separator+"model_name.bin");
        DoccatModel model = new DoccatModel(is);
        DocumentCategorizer doccat = new DocumentCategorizerME(model);
        List<String> title_list  = PrepareData.readTxtFileIntoStringArrList("data_keyword"+File.separator+"test.txt");
        String[] lt;
        double[] aProbs;
        int tp=0, fp=0, fn=0, tn=0;
        String l;
        for(String docWords : title_list) {
            lt = docWords.split("\\t+",2);
            aProbs = doccat.categorize(lt[1].split("\\s+"));
            for(int i=0; i<doccat.getNumberOfCategories(); i++){
                l = doccat.getCategory(i);
                if(lt[0].equals(l) && doccat.getBestCategory(aProbs).equals(l)){
                    tp++;
                }
                else if(lt[0].equals(l) && !doccat.getBestCategory(aProbs).equals(l)){
                    fp++;
                }
                else if(!lt[0].equals(l) && !doccat.getBestCategory(aProbs).equals(l)){
                    tn++;
                }
                else if(!lt[0].equals(l) && doccat.getBestCategory(aProbs).equals(l)){
                    fn++;
                }
            }
//            System.out.println("\n---------------------------------\nCategory : Probability\n---------------------------------");
//            System.out.println(docWords);
//            System.out.println("---------------------------------");
//            System.out.println(ap);
//            System.out.println("\n");
//            System.out.println(doccat.getAllResults(aProbs));
//            System.out.printf("\n----------%s------%s-----------------\n",dS,cat.size());

        }
        System.out.println("\n  T   F");
        System.out.printf("T %s %s\n",tp,fp);
        System.out.printf("F %s %s\n",fn,tn);
        double precison = (double) tp / (tp + fp);
        double recall = (double) tp / (tp + fn);
        double f1score = (double) 2 * precison * recall/ (precison + recall);

        System.out.printf("\nTest size: %s, precison: %.2f%%, recall: %.2f%% , f1-score: %.2f%%\n",title_list.size(), precison*100, recall*100,f1score*100);
        System.out.printf("\nAccuracy: %.2f%%\n",(double) tp / title_list.size() * 100);
    }

    private static void eval() throws Exception{
        InputStreamFactory isf = new MarkableFileInputStreamFactory(new File("data"+File.separator+"test.txt"));
        ObjectStream<String> lineStream = new PlainTextByLineStream(isf, "UTF-8");
        ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

        InputStream is = new FileInputStream("model"+File.separator+"model_name.bin");
        DoccatModel model = new DoccatModel(is);
        DocumentCategorizer doccat = new DocumentCategorizerME(model);

        List<EvaluationMonitor<DocumentSample>> listeners = new LinkedList<EvaluationMonitor<DocumentSample>>();
        listeners.add(new DoccatEvaluationErrorListener());
        listeners.add(new DoccatFineGrainedReportListener());

        DocumentCategorizerEvaluator eval =
                new DocumentCategorizerEvaluator(
                        doccat,
                        listeners.toArray(new DoccatEvaluationMonitor[listeners.size()]));

        eval.evaluate(sampleStream);
        System.out.println(eval);
    }

    private static void predict() throws Exception{
        InputStream is = new FileInputStream("model"+File.separator+"model_name.bin");
        DoccatModel model = new DoccatModel(is);
        DocumentCategorizer doccat = new DocumentCategorizerME(model);
        String buff;
        double[] aProbs;
        String cat;
        String keywords = "cosmetics";
        System.out.println("\n---------------------------------\nCategory : Probability\n---------------------------------");
            if(!StringUtils.isBlank(keywords)) {
                buff = PrepareData.preProcess_keyword(keywords);
                aProbs = doccat.categorize(buff.split("\\s+"));
                cat = doccat.getBestCategory(aProbs);
                System.out.println(doccat.getAllResults(aProbs));
            }
        System.out.println("---------------------------------");
    }

    public static void main(String[] args) throws  Exception{
//        train();
//        test_ml();
//        test();
//        predict();
        eval();


    }

}
