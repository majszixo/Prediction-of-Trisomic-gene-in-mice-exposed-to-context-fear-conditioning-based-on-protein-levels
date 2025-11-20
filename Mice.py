
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib
matplotlib.use("Agg")

import openai
import os
from dotenv import load_dotenv
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


features = [
    'DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 
    'pCAMKII_N', 'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 
    'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 
    'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 
    'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N', 'MTOR_N', 'P38_N', 
    'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 
    'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 
    'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N', 'nNOS_N', 
    'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 
    'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 
    'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N']
 
def dataPrepare(path):
    data = pd.read_csv(path, encoding = 'latin1')
    data = data.replace(",", ".", regex = True)
    #print(data.describe())
    #print(data.columns)

    target = 'Genotype'

    data[target] = data[target].replace({"Ts65Dn": "Trisomic"})

    for col in features:
        data[col] = pd.to_numeric(data[col], errors = 'coerce')

    data = data[features + [target]].dropna()

    X = data[features].to_numpy()
    y = data[target].to_numpy() 

    return data, X, y

""" def plotData(data, target):
    plt.figure(figsize = (8, 6))
    data['AvgProtein'] = data[features].mean(axis = 1)
    genotype_codes, genotype_names = pd.factorize(data[target])
    plt.scatter(genotype_codes + np.random.uniform(-0.1, 0.1, size = len(genotype_codes)),
            data['AvgProtein'], alpha = 0.7)
    plt.xticks(ticks = np.unique(genotype_codes), labels = genotype_names)
    plt.xlabel('Genotype')
    plt.ylabel('Average Protein Level')
    plt.title('Average Protein Levels by Genotype')
    plt.show()

    genotypes = data[target].unique()
    colors = ['gray', 'pink']
    for genotype, color in zip(genotypes, colors):
        subset = data[data[target] == genotype]
        plt.scatter(subset['Tau_N'], subset['ITSN1_N'], color = color, label = genotype, alpha = 0.7)
    plt.xlabel('Tau_N')
    plt.ylabel('ITSN1_N')
    plt.title('Tau_N VS ITSN1_N Protein Levels by Genotype')
    plt.legend()
    plt.show()  """

def trainTest(X, y, newData, testSize):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    newData = scaler.transform(newData.reshape(1, -1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 1234) 

    return X_train, X_test, y_train, y_test, newData



def classificationTree(randomState, X_train, y_train, X_test, y_test):
    classifierTree = DecisionTreeClassifier(random_state = randomState)
    modelTree = classifierTree.fit(X_train, y_train)

    print(f"Score Tree Model Train - {modelTree.score(X_train, y_train):.5f}") 
    print(f"Score Tree Model Test - {modelTree.score(X_test, y_test):.5f}")

    gridTree = {"max_depth": (5, 10, 15, 20),
        "min_samples_split": (5, 10, 15),
        "min_samples_leaf":(2, 5, 10)}

    gcvTree = GridSearchCV(estimator = classifierTree, param_grid = gridTree)
    gcvTree.fit(X_train, y_train)

    modelTreeGCV = gcvTree.best_estimator_
    modelTreeGCV.fit(X_train, y_train)

    print(f"Score Tree Model Train after GCV - {modelTreeGCV.score(X_train, y_train):.5f}")
    print(f"Score Tree Model Test after GCV- {modelTreeGCV.score(X_test, y_test):.5f}") 

    return modelTreeGCV

def randomForest(randomState, X_train, y_train, X_test, y_test):
    randomForest = RandomForestClassifier(random_state = randomState)
    modelRandomForest = randomForest.fit(X_train, y_train)

    print(f"Score Random Forest Model Train - {modelRandomForest.score(X_train, y_train):.5f}")
    print(f"Score Random Forest Model Test - {modelRandomForest.score(X_test, y_test):.5f}")

    gridRandomForest = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5, 10]}

    gcvRandomForest = GridSearchCV(estimator = randomForest, param_grid = gridRandomForest, cv = 5)
    gcvRandomForest.fit(X_train, y_train)
    modelRandomForestGCV = gcvRandomForest.best_estimator_

    print(f"Score Random Forest Model Train after GCV - {modelRandomForestGCV.score(X_train, y_train):.5f}")
    print(f"Score Random Forest Model Test after GCV - {modelRandomForestGCV.score(X_test, y_test):.5f}")
    return modelRandomForestGCV

def gradientBoosting(randomState, learningRate, X_train, y_train, X_test, y_test):

    gradientBoosting = GradientBoostingClassifier( random_state = randomState, learning_rate = learningRate)
    modelGradientBoosting = gradientBoosting.fit(X_train, y_train)

    print(f"Score Gradient Boosting Model Train - {modelGradientBoosting.score(X_train, y_train):.5f}")
    print(f"Score Gradient Boosting Model Test - {modelGradientBoosting.score(X_test, y_test):.5f}")
    return modelGradientBoosting


def predict(model, newData):
    prediction = model.predict(newData)
    print(f"Prediction for new data with the use of {model} - {prediction}")


def evaluate(models : dict, X, y, cv):

    scoring_metrics = {
        "Accuracy": "accuracy",
        "Recall": "recall_macro",
        "Precision": "precision_macro",
        "F1-score": "f1_macro"
    }

    for model_name, model in models.items():
        print(f"Evaluation for {model_name}")
        for metric_name, scoring in scoring_metrics.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            avg_score = np.mean(scores)
            print(f"{metric_name}: {', '.join(f'{score:.5f}' for score in scores)}")
            print(f"Average {metric_name}: {avg_score:.5f}")



def recFeatureElimination(est, X_train, y_train, X_test, newData, n_features, steps):
    if newData.ndim == 1:
        newData = newData.reshape(1, -1)
    rfeSelector = RFE(estimator=est, n_features_to_select=n_features, step=steps)
    rfeSelector.fit(X_train, y_train)
    selected_features = [feature for feature, selected in zip(features, rfeSelector.support_) if selected]
    print(f"Selected Features: {selected_features}")

    X_train_rfe = rfeSelector.transform(X_train)
    X_test_rfe = rfeSelector.transform(X_test)
    newData = rfeSelector.transform(newData)
    return rfeSelector, X_train_rfe, X_test_rfe, newData


def compareModels(X_train, y_train, X_test, y_test, cv=5, random_state=42, learning_rate=0.1):
    models = {
        "Decision Tree (GCV)": classificationTree(random_state, X_train, y_train, X_test, y_test),
        "Random Forest (GCV)": randomForest(random_state, X_train, y_train, X_test, y_test),
        "Gradient Boosting": gradientBoosting(random_state, learning_rate, X_train, y_train, X_test, y_test)
    }

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = f1
        print(f"{name} - Test F1-score: {f1:.5f}")

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} (F1-score = {results[best_model_name]:.5f})")
    return best_model

def confusionMatrix(y_true, y_pred, labels = None, title = "Confusion Matrix", cmap = "RdPu"):
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    disp.plot(cmap = cmap)
    plt.title(title)
    plt.savefig(r"C:\Users\agnie\Desktop\ML\confusion_matrix.png")
    plt.close()
    print(classification_report(y_true, y_pred, target_names=labels))

def generateMouseChatGPT():
    prompt = (
        "Generate a new set of exactly 77 protein expression levels from a mouse brain sample. "
        "Return only a comma-separated list of 77 float values, one line, no explanation. "
        "Values must be realistic (between 0.0 and 6.0)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates numerical data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        reply = response.choices[0].message.content.strip()
        values = [float(x.strip()) for x in reply.replace("\n", "").split(",") if x.strip()]

        if len(values) < 77:
            raise ValueError(f"Expected 77 values, but got {len(values)}")
        elif len(values) > 77:
            print(f"Warning: Got {len(values)} values, trimming to 77")
            values = values[:77]

        print("Generated Mouse Data (from ChatGPT):")
        print(values)
        return np.array(values)

    except Exception as e:
        print(f"Error generating data from ChatGPT: {e}")
        return None

def main():

    newData = generateMouseChatGPT()
    if newData is None:
        print("Failed to generate new data")
        return


    path = r"C:\Users\agnie\Desktop\ML\datacortex.csv" 
    data, X, y = dataPrepare(path)
    

    X_train, X_test, y_train, y_test, newData_scaled = trainTest(X, y, newData, testSize=0.2)

    rfe_selector, X_train_rfe, X_test_rfe, newData_rfe = recFeatureElimination(
        RandomForestClassifier(random_state=42), 
        X_train, y_train, X_test, newData_scaled, 
        n_features=30, steps=5)
    
    print("\nModel Comparison")
    best_model = compareModels(X_train_rfe, y_train, X_test_rfe, y_test)
    

    print("\nCross-Validation Evaluation")
    models = {
        "Decision Tree": classificationTree(42, X_train_rfe, y_train, X_test_rfe, y_test),
        "Random Forest": randomForest(42, X_train_rfe, y_train, X_test_rfe, y_test),
        "Gradient Boosting": gradientBoosting(42, 0.1, X_train_rfe, y_train, X_test_rfe, y_test)
    }
    evaluate(models, X_train_rfe, y_train, cv=5)

    print("\nTest Set Performance")
    y_pred = best_model.predict(X_test_rfe)
    confusionMatrix(y_test, y_pred, labels=np.unique(y))
    

    print("\nNew Data Prediction")
    predict(best_model, newData_rfe)
    


if __name__ == "__main__":
    main()