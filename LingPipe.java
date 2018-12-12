import com.aliasi.classify.*;
import com.aliasi.classify.NaiveBayesClassifier;
import com.aliasi.lm.TokenizedLM;
import com.aliasi.tokenizer.NGramTokenizerFactory;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Strings;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class LanguageClassifierLingPipe {

    private static String[] cat_train={"t1", "t2"};

    private static void gridSearch() throws Exception{
        Double maxScore = 0.0;
        File dataDir = new File("data"+File.separator+"train");
        int maxi=0, maxj =0, maxk=0;
        List<String> title_list  = PrepareData.readTxtFileIntoStringArrList("data"+File.separator+"test_sample_1.txt");

        for(int i=2;i<6;i++) {
            for(int j=i+1;j<8;j++) {
                for(int k=0;k<5;k++) {
//                DynamicLMClassifier classifier =
//                        new BinaryLMClassifier(
//                                new NGramProcessLM(j),
//                                Math.pow(10, i),
//                                "other",
//                                "en");
//                DynamicLMClassifier<NGramProcessLM> classifier
//                = DynamicLMClassifier
//                .createNGramProcess(cat_train,j);
                    TokenizerFactory factory
                            = new NGramTokenizerFactory(i, j);
//                    System.out.println("maxnGram:" + i + ", minnGram:" + j + ", charSmoothingNGram:" + k);

                    DynamicLMClassifier<TokenizedLM> classifier =
                            new NaiveBayesClassifier(cat_train,
                                    factory, k);

                    for (String filename : dataDir.list()) {
                        File trainingFile = new File(dataDir, filename);
                        FileInputStream fileIn
                                = new FileInputStream(trainingFile);
                        InputStreamReader reader
                                = new InputStreamReader(fileIn, Strings.UTF8);
                        BufferedReader bufferedReader = new BufferedReader(reader);
                        String lineTxt;
                        String[] lt;
                        while ((lineTxt = bufferedReader.readLine()) != null) {
                            lt = lineTxt.split("\t", 2);
                            Classification c = new Classification(lt[0]);
                            if (!StringUtils.isBlank(lt[1])) {
                                Classified<CharSequence> classified = new Classified<>(lt[1], c);
                                classifier.handle(classified);
                            }
                        }
                        reader.close();
                    }

                    String category;
                    int[][] confusion_matrix = new int[2][2];
                    for (String lineTxt : title_list) {
                        String[] lt = lineTxt.split(",", 2);
                        String title = PrepareData.preProcess_language(lt[1]);
                        String category = classification.bestCategory(title); 

                        if(category.equals(lt[0]) && lt[0].equals("t1")){
                            confusion_matrix[0][0]++;
                        }
                        else if(category.equals(lt[0]) && !cat.equals("t1")) {
                            confusion_matrix[1][1]++;
                        }
                        else if(category.equals(lt[0]) && cat.equals("t1")) {
                            confusion_matrix[0][1]++;
                        }
                        else if(category.equals(lt[0]) && !cat.equals("t1")) {
                            confusion_matrix[1][0]++;
                        }
                   

                        double score = (double) (confusion_matrix[0][0] + confusion_matrix[1][1]) / title_list.size();
                        if (score > maxScore) {
                            maxScore = score;
                            maxi = i;
                            maxj = j;
                            maxk = k;
                        }
                        System.out.println("minNgram= " + i + ", MaxNgram= " + j + ", charSmoothing= " + k +", score=" + score);
    //                System.out.println("ngram= "+j+", score="+ score);
                }
            }
        }
        System.out.println("minNgram="+ maxi +", MaxNgram="+ maxj +", maxCharSmoothing= " + maxk +", maxScore="+ maxScore);
    //minNgram=4, MaxNgram=7, maxCharSmoothing= 3, maxScore=0.92
    }

    private static void train() throws Exception {
        File dataDir = new File("data"+File.separator+"train");
        File modelFile = new File("model"+File.separator+"lingpipe_model.bin");
        int maxnGram = 7;
        int minnGram = 4;
        int charSmoothingNGram= 3;
        TokenizerFactory factory
                = new NGramTokenizerFactory(minnGram,maxnGram);
        System.out.println("maxnGram:"+maxnGram+", minnGram:"+minnGram+", charSmoothingNGram:"+charSmoothingNGram);

        DynamicLMClassifier<TokenizedLM> classifier =
                new NaiveBayesClassifier(cat_train,
                    factory, charSmoothingNGram);

//        DynamicLMClassifier<NGramProcessLM> classifier
//                = DynamicLMClassifier
//                .createNGramProcess(cat_train,3);
//        DynamicLMClassifier classifier=
//                new BinaryLMClassifier(
//                        new NGramProcessLM(maxnGram),
//                        1e1,
//                        "t1",
//                        "t2");
        for(String filename:dataDir.list()) {
            File trainingFile = new File(dataDir, filename);
            FileInputStream fileIn
                    = new FileInputStream(trainingFile);
            InputStreamReader reader
                    = new InputStreamReader(fileIn, Strings.UTF8);
            BufferedReader bufferedReader = new BufferedReader(reader);
            String lineTxt;
            String[] lt;
            while ((lineTxt = bufferedReader.readLine()) != null) {
                lt = lineTxt.split("\t", 2);
                Classification c = new Classification(lt[0]);
                if (!StringUtils.isBlank(lt[1])) {
                    Classified<CharSequence> classified = new Classified<>(lt[1], c);
                    classifier.handle(classified);
                }
            }
            reader.close();
        }
//        for(String lang:cat_train)
//            classifier.languageModel(lang).substringCounter().prune(5);

        AbstractExternalizable.compileTo(classifier,modelFile);
        System.out.println("\nCompiling model to file=" + modelFile);
    }

    

    private static void evaluate() throws Exception{
        File dataDir = new File("data_language");
        File modelFile = new File("model"+File.separator+"lingpipe_model.bin");

        BaseClassifier<CharSequence> classifier
                = (BaseClassifier<CharSequence>) AbstractExternalizable.readObject(modelFile);
        BaseClassifierEvaluator<CharSequence> evaluator
                = new BaseClassifierEvaluator<>(classifier, cat_train,false);

        String[] file_test ={"test_sample_1", "test_sample_2"};
        
        String[] lt;
        String title;
        for(String file : file_test) {
            File testingFile = new File(dataDir,
                    file+".txt");
            FileInputStream fileIn
                    = new FileInputStream(testingFile);
            InputStreamReader reader
                    = new InputStreamReader(fileIn, Strings.UTF8);

            BufferedReader bufferedReader = new BufferedReader(reader);
            String lineTxt = null;
            while ((lineTxt = bufferedReader.readLine()) != null) {
                lt = lineTxt.split(",",2);
                title = PrepareData.preProcess_language(lt[1]);
                Classification c = new Classification(lt[0]);
                if(!StringUtils.isBlank(title)) {
                    Classified<CharSequence> cl
                            = new Classified<>(title, c);
                    evaluator.handle(cl);
                }
            }
        }
        System.out.println(evaluator.toString());
    }

    private static void test() throws Exception {
    
        File dataDir = new File("data");
        File modelFile = new File("model"+File.separator+"lingpipe_model.bin");
        BaseClassifier<CharSequence> classifier
                = (BaseClassifier<CharSequence>) AbstractExternalizable.readObject(modelFile);
                
        int[][] confusion_matrix=new int[2][2];
        
        List<String> title_list  = PrepareData.readTxtFileIntoStringArrList("data"+File.separator+"test_sample_1.txt");
//        title_list.addAll(readTxtFileIntoStringArrList("data"+File.separator+"test_sample_2.txt"));

        for(String lineTxt : title_list) {
                String[] lt = lineTxt.split(",",2);
                
                String s=PrepareData.preProcess(lt[1]);
                Classification classification
                            = classifier.classify(s);
                String category = classification.bestCategory(); 

                    if(category.equals(lt[0]) && lt[0].equals("t1")){
                        confusion_matrix[0][0]++;
                    }
                    else if(category.equals(lt[0]) && !cat.equals("t1")) {
                        confusion_matrix[1][1]++;
                    }
                    else if(category.equals(lt[0]) && cat.equals("t1")) {
                        confusion_matrix[0][1]++;
                    }
                    else if(category.equals(lt[0]) && !cat.equals("t1")) {
                        confusion_matrix[1][0]++;
                    }
            }

        double res =(double) (confusion_matrix[0][0]+confusion_matrix[1][1])/title_list.size();
        System.out.printf("Test size: %s, accuracy: %.2f %%\n",title_list.size(), res*100);
        System.out.println("confusion matrix");
        System.out.println("       en,   other");
        System.out.printf("en:    %s,   %s\n",confusion_matrix[0][0],confusion_matrix[0][1]);
        System.out.printf("other: %s,   %s\n",confusion_matrix[1][0],confusion_matrix[1][1]);
        System.out.printf("accuracy en: %.2f %%\n",(double)confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])*100);
        System.out.printf("accuracy other: %.2f %%\n",(double)confusion_matrix[1][1]/(confusion_matrix[1][0]+confusion_matrix[1][1])*100);
        System.out.printf("size_en: %s size_other: %s\n",size_en,size_other);
    }
    
    private static void predict(String[] query) throws Exception{
        File modelFile = new File("model"+File.separator+"lingpipe_model.bin");
        BaseClassifier<CharSequence> classifier
                = (BaseClassifier<CharSequence>) AbstractExternalizable.readObject(modelFile);
        for(String s :query) {
            String process = PrepareData.preProcess_language(s);
            System.out.println(s);
            System.out.println(process);
            System.out.println(classifier.classify(process).bestCategory());
            System.out.println(classifier.classify(process));
        }
    }

    private static void ngram(String query) throws Exception{
        int maxnGram = 8;
        int minnGram = 3;
        TokenizerFactory factory
                = new NGramTokenizerFactory(minnGram,maxnGram);
        String[] s = factory.tokenizer(query.toCharArray(),0,query.length()).tokenize();
        System.out.println(Arrays.stream(s).collect(Collectors.joining(" \n")));
    }

    public static void main(String[] args) throws  Exception{
//        train();
//        test();
        gridSearch();
//        evaluate();
//        predict(queries);
    }
}
