import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class ActivityStatePredictor:
    def __init__(self, k_folds=5):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.is_trained = False
        self.feature_names = ['Steps', 'Calories_Out']
        self.class_names = ['Poco movimento', 'Discreto movimento', 'Movimento ottimale']
        self.k_folds = k_folds
        self.cv_results = None

    def load_data(self, train_file, test_file=None):

        try:

            self.train_data = pd.read_csv(train_file)
            print(f"Dataset di training caricato: {self.train_data.shape[0]} campioni")
            print(f"Colonne: {list(self.train_data.columns)}")


            required_cols = ['Steps', 'Calories_Out', 'State']
            missing_cols = [col for col in required_cols if col not in self.train_data.columns]
            if missing_cols:
                raise ValueError(f"Colonne mancanti nel dataset di training: {missing_cols}")


            self.X_train = self.train_data[self.feature_names]
            self.y_train = self.train_data['State']

            print(f"Distribuzione classi nel training set:")
            print(self.y_train.value_counts().sort_index())


            if test_file and os.path.exists(test_file):
                self.test_data = pd.read_csv(test_file)
                print(f"Dataset di test caricato: {self.test_data.shape[0]} campioni")


                if 'State' in self.test_data.columns:
                    self.X_test = self.test_data[self.feature_names]
                    self.y_test = self.test_data['State']
                else:
                    self.X_test = self.test_data[self.feature_names]
                    self.y_test = None
                    print("Il dataset di test non contiene la colonna 'State' - solo predizione")
            else:
                print("Nessun dataset di test fornito - sarà creato uno split dal training set")
                self.X_test = None
                self.y_test = None

        except Exception as e:
            print(f"Errore nel caricamento dei dati: {e}")
            raise

    def perform_cross_validation(self):
        try:
            print(f"\n=== CROSS VALIDATION ({self.k_folds}-fold) ===")

            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

            cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

            cv_results = cross_validate(
                self.model, self.X_train, self.y_train,
                cv=cv, scoring=scoring, return_train_score=True
            )

            self.cv_results = cv_results

            print("\nRisultati per ogni fold:")
            print("-" * 80)
            print(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("-" * 80)

            for i in range(self.k_folds):
                print(f"{i + 1:<6} "
                      f"{cv_results['test_accuracy'][i]:<12.4f} "
                      f"{cv_results['test_precision_macro'][i]:<12.4f} "
                      f"{cv_results['test_recall_macro'][i]:<12.4f} "
                      f"{cv_results['test_f1_macro'][i]:<12.4f}")

            print("-" * 80)

            print("\nMedia dei risultati cross validation:")
            accuracy_mean = np.mean(cv_results['test_accuracy'])
            precision_mean = np.mean(cv_results['test_precision_macro'])
            recall_mean = np.mean(cv_results['test_recall_macro'])
            f1_mean = np.mean(cv_results['test_f1_macro'])

            print(f"Accuracy media: {accuracy_mean:.4f}")
            print(f"Precision media: {precision_mean:.4f}")
            print(f"Recall media: {recall_mean:.4f}")
            print(f"F1-Score media: {f1_mean:.4f}")

            return cv_results

        except Exception as e:
            print(f"Errore durante la cross validation: {e}")
            raise

    def train_model(self):
        try:
            self.perform_cross_validation()

            if self.X_test is None:
                self.X_train_final, self.X_test, self.y_train_final, self.y_test = train_test_split(
                    self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
                )
                print("\nCreato split train/test (80/20) per valutazione finale")
            else:
                self.X_train_final = self.X_train
                self.y_train_final = self.y_train

            print("\n=== TRAINING FINALE DEL MODELLO ===")
            print("Inizio training del modello Random Forest sull'intero training set...")

            self.model.fit(self.X_train_final, self.y_train_final)
            self.is_trained = True

            if self.y_test is not None:
                y_pred = self.model.predict(self.X_test)

                test_accuracy = accuracy_score(self.y_test, y_pred)
                test_precision = precision_score(self.y_test, y_pred, average='macro')
                test_recall = recall_score(self.y_test, y_pred, average='macro')
                test_f1 = f1_score(self.y_test, y_pred, average='macro')

                print(f"\nRisultati sul test set:")
                print(f"Accuracy: {test_accuracy:.4f}")
                print(f"Precision: {test_precision:.4f}")
                print(f"Recall: {test_recall:.4f}")
                print(f"F1-Score: {test_f1:.4f}")

            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nImportanza delle features:")
            print(feature_importance)

        except Exception as e:
            print(f"Errore durante il training: {e}")
            raise

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Il modello deve essere addestrato prima di fare predizioni")

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def plot_confusion_matrix(self, save_path=None):
        if not self.is_trained or self.y_test is None:
            raise ValueError("Il modello deve essere addestrato e avere dati di test con labels")

        y_pred = self.model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Matrice di Confusione - Random Forest')
        plt.xlabel('Predizione')
        plt.ylabel('Valore Reale')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matrice di confusione salvata in: {save_path}")

        plt.show()

        print("\nReport di Classificazione:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))

        return cm

    def save_model(self, model_path):
        if not self.is_trained:
            raise ValueError("Il modello deve essere addestrato prima di essere salvato")

        joblib.dump(self.model, model_path)
        print(f"Modello salvato in: {model_path}")

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"Modello caricato da: {model_path}")

    def predict_from_data(self, data):
        if isinstance(data, list):
            data = np.array(data).reshape(1, -1)
        elif isinstance(data, dict):
            data = pd.DataFrame([data])

        if isinstance(data, pd.DataFrame):
            data = data[self.feature_names]

        predictions, probabilities = self.predict(data)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'predizione': pred,
                'classe': self.class_names[pred],
                'probabilita': {
                    self.class_names[j]: prob[j] for j in range(len(self.class_names))
                }
            }
            results.append(result)

        return results

    def get_cv_summary(self):
        if self.cv_results is None:
            print("Cross validation non ancora eseguita")
            return None

        summary = {
            'accuracy_mean': np.mean(self.cv_results['test_accuracy']),
            'accuracy_std': np.std(self.cv_results['test_accuracy']),
            'precision_mean': np.mean(self.cv_results['test_precision_macro']),
            'precision_std': np.std(self.cv_results['test_precision_macro']),
            'recall_mean': np.mean(self.cv_results['test_recall_macro']),
            'recall_std': np.std(self.cv_results['test_recall_macro']),
            'f1_mean': np.mean(self.cv_results['test_f1_macro']),
            'f1_std': np.std(self.cv_results['test_f1_macro'])
        }

        return summary


def main():
    predictor = ActivityStatePredictor(k_folds=5)

    train_file = "training.csv"
    test_file = "test.csv"
    model_path = "random_forest_model.pkl"

    try:
        predictor.load_data(train_file, test_file)

        predictor.train_model()

        predictor.save_model(model_path)

        if predictor.y_test is not None:
            predictor.plot_confusion_matrix("confusion_matrix.png")


    except Exception as e:
        print(f"Errore nell'esecuzione: {e}")


def load_and_predict(model_path, new_data):
    predictor = ActivityStatePredictor()
    predictor.load_model(model_path)

    results = predictor.predict_from_data(new_data)
    return results


if __name__ == "__main__":
    main()