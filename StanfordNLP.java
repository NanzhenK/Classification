import edu.stanford.nlp.classify.ColumnDataClassifier;
import edu.stanford.nlp.ling.Datum;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.*;
import java.io.File;
import java.io.*;
import java.util.List;
import java.util.regex.Pattern;

public class StanfordNLP {
    private static void train() throws Exception {
            ColumnDataClassifier classifier = new ColumnDataClassifier("train_nb.properties");
            classifier.trainClassifier("data_language"+File.separator+"train.txt");
    //        System.out.println(classifier.testClassifier("test.txt"));
        }

        private static void test() throws Exception {
            ColumnDataClassifier classifier = ColumnDataClassifier.getClassifier("classify.model");
    //            System.out.println(classifier.testClassifier("test.txt"));
            PrintWriter writer = new PrintWriter("res.txt");
            List<String> predict_list = readTxtFileIntoStringArrList("test.txt");
            for (String line : predict_list) {
                Datum<String,String> d = classifier.makeDatumFromLine(line);
                writer.printf("%s  ==>  %s (%.4f)%n \n", line, classifier.classOf(d), classifier.scoresOf(d).getCount(classifier.classOf(d)));
            }
            writer.flush();
            writer.close();
        }

        public static void main(String args[]) throws Exception  {
            train();
           test();
        }
}
