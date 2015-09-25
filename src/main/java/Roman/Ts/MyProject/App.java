package Roman.Ts.MyProject;
import java.util.Arrays;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class App {
	//Тестую приклад з сайту http://bazhenov.me/blog/2012/06/11/naive-bayes.html
    public static void main(String[] args) {
    	//Використовую свою програму методу Байєса
    	BayesClassifier<String, String> bayes = new BayesClassifier<String, String>();
		String[] spam = "предоставляю услуги бухгалтера".toLowerCase().split("\\s+|,\\s*|\\.\\s*");
		//Передаю повідомлення, відмічене як "спам"
		bayes.learn("спам", Arrays.asList(spam));
		String[] spam2 = "спешите купить виагру".toLowerCase().split("\\s+|,\\s*|\\.\\s*");
		//Передаю повідомлення, відмічене як "спам"
		bayes.learn("спам", Arrays.asList(spam2));
		String[] ham = "надо купить молоко".toLowerCase().split("\\s+|,\\s*|\\.\\s*");
		//Передаю повідомлення, відмічене як "не спам"
		bayes.learn("не спам", Arrays.asList(ham));
		String[] unknownText = "надо купить хлеб".toLowerCase().split("\\s+|,\\s*|\\.\\s*");	
		//Передаю повідомлення і визначаю його клас
        String myResult = bayes.classify(Arrays.asList(unknownText));
		
		//Тестую ті ж дані на бібліотеці MlLib від Apache Spark
    	SparkConf conf = new SparkConf().setMaster("local").setAppName("My app");
        JavaSparkContext context = new JavaSparkContext(conf);
        LibraryMetods app = new LibraryMetods();
        //Записую повідомлення у вигляді вектора, в якому вказую кількість вживань
        //кожного терміна з мішка слів для навчання
		Vector testData = Vectors.dense(new double[]{0, 0, 0, 0, 1, 0, 1, 0});
		//Визначаю клас бібліотечним методом Байєса
        String result1 = app.naiveBayes(context, testData) == 0 ? "спам" : "не спам";
		//Визначаю клас бібліотечним методом дерева рішень
        String result2 = app.classificationDecisionTree(context, testData) == 0 ? "спам" : "не спам";
		//Визначаю клас бібліотечним методом опорних векторів
        String result3 = app.supportVectorMachine(context, testData) == 0 ? "спам" : "не спам";
        //Виводжу результати
		System.out.println("Моя реалізація методу Байєса: " + myResult);	
        System.out.println("Бібліотечний метод Байєса: " + result1);
        System.out.println("Бібліотечний метод дерева рішень: " + result2);
        System.out.println("Бібліотечний метод опорних векторів: " + result3);
    }
}