from model_randomForest import load_and_predict
import pandas as pd

csv_file = "mydata.csv"
df = pd.read_csv(csv_file)

results = load_and_predict("random_forest_model.pkl", df)

print(f"Risultati inferenza:")
for i, result in enumerate(results[:]):
    steps = df.iloc[i]['Steps']
    calories = df.iloc[i]['Calories_Out']
    print(f"  Campione {i + 1}: Steps={steps}, Calories={calories}")
    print(f"    → {result['classe']} (prob: {max(result['probabilita'].values()):.3f})")
