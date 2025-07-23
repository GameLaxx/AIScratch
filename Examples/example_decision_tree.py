from AIScratch.DecisionTree import DecisionTree, GiniPurity
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Examples/titanic.csv")
df_filtered = df.dropna()
train_df, test_df = train_test_split(df_filtered, test_size=0.3, random_state=42)
gini = GiniPurity()
tree = DecisionTree(gini, max_depth=6)
tree.build_tree(train_df, "Survived")
print("Accuracy:", tree.evaluate(test_df, "Survived"))
