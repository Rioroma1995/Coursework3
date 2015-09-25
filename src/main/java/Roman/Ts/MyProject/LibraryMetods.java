package Roman.Ts.MyProject;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;

public class LibraryMetods {
	public double supportVectorMachine(JavaSparkContext context, Vector testData) {
		//Створюю вектори з даних для подальшого навчання
        RDD<LabeledPoint> data = MLUtils.loadLabeledData(context.sc(), "train.data");
        //Рекомендована розробниками к-ть ітерацій
		int numIterations = 100;
		//Передаю дані для навчання і отримую вже навчену модель
		SVMModel model = SVMWithSGD.train(data, numIterations);
		//Визначаю клас тестового прикладу
	    return model.predict(testData);
	}
    public double naiveBayes(JavaSparkContext context, Vector testData) {
		//Створюю вектори з даних для подальшого навчання
        RDD<LabeledPoint> data = MLUtils.loadLabeledData(context.sc(), "train.data");
		//Передаю дані для навчання і отримую вже навчену модель
        NaiveBayesModel trained = NaiveBayes.train(data);
		//Визначаю клас тестового прикладу
        return trained.predict(testData);
    }
    public double classificationDecisionTree(JavaSparkContext context, Vector testData) {
		//Створюю вектори з даних для подальшого навчання
		JavaRDD<LabeledPoint> data = MLUtils.loadLabeledData(context.sc(), "train.data").toJavaRDD();
		Integer numClasses = 2;//Вказуємо к-ть класів
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		//Рекомендоване розробниками значення для обрахунку коеф. посилення інформації
		String impurity = "gini";
		Integer maxDepth = 5;//Рекомендована розробниками глибина дерева
		Integer maxBins = 32;//Рекомендована розробниками к-ть ознак
		//Передаю дані для навчання і отримую вже навчену модель
		DecisionTreeModel model = DecisionTree.trainClassifier(data, numClasses, categoricalFeaturesInfo, impurity,	maxDepth, maxBins);
		//Визначаю клас тестового прикладу
		return model.predict(testData);
	}    	
}