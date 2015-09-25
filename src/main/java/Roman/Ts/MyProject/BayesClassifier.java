package Roman.Ts.MyProject;
import java.util.Collection;
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Set;
public class BayesClassifier<Features, Category>{
	//Список усіх категорій, де для кожної категорії ставиться у відповідність список ознак, 
	//які у ній зустрілись, та кількість появи цих ознак у даній категорії
	private Dictionary<Category, Dictionary<Features, Integer>> featureCountPerCategory;
	//список усіх ознак та кількість їх зустрічі при навчання
	private Dictionary<Features, Integer> totalFeatureCount;
	//список усіх категорій та кількість їх зустрічі при навчання
	private Dictionary<Category, Integer> totalCategoryCount;
	public BayesClassifier() {
		featureCountPerCategory = new Hashtable<Category, Dictionary<Features, Integer>>();
		totalFeatureCount = new Hashtable<Features, Integer>();
		totalCategoryCount = new Hashtable<Category, Integer>();
	}
	//Повертаю кількість унікальних ознак
	public int getFeaturesCount() {
		return totalFeatureCount.size();
	}
	//Повертаю усі ознаки
	public Set<Features> getFeatures() {
		return ((Hashtable<Features, Integer>) totalFeatureCount).keySet();
	}
	//Повертаю усі категорії
	public Set<Category> getCategories() {
		return ((Hashtable<Category, Integer>) totalCategoryCount).keySet();
	}
	//Повертаю загальну кількість появи всіх категорій
	public int getCategoriesTotal() {
		int toReturn = 0;
		for (Enumeration<Integer> e = totalCategoryCount.elements(); e.hasMoreElements();)
			toReturn += e.nextElement();
		return toReturn;
	}
	//Повертаю загальну кількість ознак з категорії
	public int getFeaturesCountInCategory(Category category) {
		int toReturn = 0;
		Dictionary<Features, Integer> features = featureCountPerCategory.get(category);
		for (Enumeration<Integer> e = features.elements(); e.hasMoreElements();)
			toReturn += e.nextElement();
		return toReturn;
	}
	//Додаю нову ознаку до категорії або збільшую кількість її появи у цій категорії, 
	//якщо вона вже раніше зустрічалась у даній категорії
	public void incrementFeature(Features feature, Category category) {
		Dictionary<Features, Integer> features = featureCountPerCategory.get(category);
		if (features == null) {
			featureCountPerCategory.put(category, new Hashtable<Features, Integer>());
			features = featureCountPerCategory.get(category);
		}
		Integer count = features.get(feature);
		if (count == null) {
			features.put(feature, 0);
			count = features.get(feature);
		}
		features.put(feature, ++count);
		Integer totalCount = totalFeatureCount.get(feature);
		if (totalCount == null) {
			totalFeatureCount.put(feature, 0);
			totalCount = totalFeatureCount.get(feature);
		}
		totalFeatureCount.put(feature, ++totalCount);
	}
	//Додаю нову категорію, або збільшую кількість її зустрічі при навчанні, 
	//якщо вона вже зустріччалась до цього моменту
	public void incrementCategory(Category category) {
		Integer count = totalCategoryCount.get(category);
		if (count == null) {
			totalCategoryCount.put(category, 0);
			count = totalCategoryCount.get(category);
		}
		totalCategoryCount.put(category, ++count);
	}
	//Кількість зустрічі даної ознаки у певній категорії
	public int featureCount(Features feature, Category category) {
		Dictionary<Features, Integer> features = featureCountPerCategory.get(category);
		if (features == null)
			return 0;
		Integer count = features.get(feature);
		return (count == null) ? 0 : count.intValue();
	}
	//Кількість зустрічі певної категорії при навчанні
	public int categoryCount(Category category) {
		Integer count = totalCategoryCount.get(category);
		return (count == null) ? 0 : count.intValue();
	}
	//навчаємо модель
	public void learn(Category category, Collection<Features> features) {
		//Збільшую зустріч ознаки в цій категорії
		for (Features feature : features)
			incrementFeature(feature, category);
		//Збільшую кількість зустрічі даної категорій при навчанні
		incrementCategory(category);
	}
	//Знаходжу умовну ймовірність 
	//log(p(x|y))=sum(log((1 + к-ть зустрічі ознаки в класі)/(к-ть унікальних ознак + заг. к-ть всіх ознак в класі)))
	private double featuresProbabilityProduct(Collection<Features> features, Category category) {
		double product = 0;
        for (Features feature : features)
            product += Math.log((1+featureCount(feature, category))/(double)(getFeaturesCountInCategory(category)+getFeaturesCount()));
        return product;
    }
	//Застосовую формулу Байєса p(y|x)=log(p(y))+log(p(x|y)) при наївній класифікації
    private double categoryProbability(Collection<Features> features, Category category) {
    	return Math.log(( categoryCount(category) / (double)getCategoriesTotal())) + featuresProbabilityProduct(features, category);
    }
    //Знаходимо категорію, що дає найбільшу умовну імовірність за формулою Байєса, 
    //що даний набір ознак належить до цього класу
    private Category categoryProbabilities(Collection<Features> features) {
        double prob = Double.NEGATIVE_INFINITY;
        Category cat = null;
        for (Category category : getCategories()){
        	double newProbability = categoryProbability(features, category);
        	System.out.println(newProbability);
        	if(newProbability>=prob){
        		prob = newProbability;
        		cat = category;  
        	}
        }
        return cat;
    }
    //Класифікуємо вхідні тестові дані і отримуємо найімовірнішу категорію
    public Category classify(Collection<Features> features) {
        return categoryProbabilities(features);
    }
}