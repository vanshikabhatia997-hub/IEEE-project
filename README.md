**PUBLIC WEBSITE ON RENDOR** : https://ieee-project-5.onrender.com/
# 🌸 Iris Flower Classification

## Project Overview
**What does the model predict?**
It predicts the **species** of an iris flower (*setosa*, *versicolor* or *virginica*) from four measurements: sepal length, sepal width, petal length and petal width (all in cm).

**Basic idea:**
In this project, I built a model that guesses which type of iris flower
it is: setosa, versicolor or virginica. It only looks at four
measurements: the **length** and **width** of the **petals**, and the **length** and
**width** of the **sepals**.

**Why I choose this?**
The same approach has **real-world value**. Predicting a category from
measurements is the core idea behind plant-identification apps, medical
screening tools and automated quality control.


## Dataset
- **Iris dataset** (UCI Machine Learning Repository): https://archive.ics.uci.edu/dataset/53/iris
- 150 flowers, 4 numeric features, 3 classes (50 each)
- No missing values
- Loaded directly in the notebook with `sklearn.datasets.load_iris()`, so no download is needed

## Setup Instructions
1. Clone this repository and open a terminal in its folder.
2. Install the requirements:
   ```bash
   pip install pandas scikit-learn matplotlib jupyter
   ```
3. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook iris_classification.ipynb
   ```
4. Choose **Run → Run All Cells**.

*Alternative:* upload the notebook to [Google Colab](https://colab.research.google.com) and run it there with nothing to install.

## Method
1. Load the data and check for missing values (none found).
2. Split features (X) and target (y), then split into 80% training / 20% test (`random_state=42`, stratified).
3. Train a Decision Tree and a KNN classifier (k = 5).
4. Evaluate on the test set with accuracy and confusion matrices; confirm with 5-fold cross-validation.

## Results

| Model | Test accuracy (30 flowers) | 5-fold cross-validation accuracy |
|---------------|-----------|-----------|
| Decision Tree | 93.33% | 95.33% |
| KNN (k=5) | **100%** | **97.33%** |

The Decision Tree's only errors were between *versicolor* and *virginica*. KNN classified every test flower correctly.

## Conclusion
**Which model performed better?**
**KNN (k=5)** performed better. It scored **100%** on the test set (30/30 correct), while the **Decision Tree** scored **93.33%** (28/30 correct). The Decision Tree's two mistakes were between *versicolor* and *virginica*. Cross-validation agrees: KNN averaged about 97.3% vs. about 95.3% for the Decision Tree, so the gap is real but small.

**Validation:** To make sure this was not a lucky split, I confirmed
the result with 5-fold cross-validation. KNN again came out ahead
(97.33% vs 95.33%) and was also more consistent (±2.49% vs ±3.40%).
The advantage is real but modest, since both models performed strongly.


**What I learned**
- Machine learning follows a simple workflow: load data → check for problems → split → train → evaluate.
- We split the data into train and test sets so we can check the model on flowers it has never seen.
- Accuracy is one number; the **confusion matrix** shows *where* a model goes wrong. Setosa is easy, while versicolor and virginica are the species that get confused.
- With a test set of only 30 flowers, one split can be misleading, so cross-validation gives a more trustworthy comparison.
- Simple models can work very well on clean, small datasets.

**Future steps:** I would tune each model and apply the same workflow to a
more complex dataset such as the Mushroom dataset, which requires encoding categorical features.

## Files
- `iris_classification.ipynb`: the full notebook with outputs
- `README.md`: this file
To test the model: streamlit run app.py 
