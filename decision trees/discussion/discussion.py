import pandas as pd
from IPython.display import Image
import pydotplus
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.Series(iris['target'])
for i in range(3): y.replace(i, iris['target_names'][i], inplace=True)

model = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=1234)
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True, ascending=False)

IMPORTANCE=pd.DataFrame(columns=['importance','feature'])
IMPORTANCE['importance']=model.feature_importances_
IMPORTANCE['feature']=pd.DataFrame(iris.feature_names)
IMPORTANCE.sort_values(by=['importance'],ascending=False)
plt.bar(IMPORTANCE['feature'],IMPORTANCE['importance'],width=0.5)
plt.xlabel('Feature')
plt.ylabel('Importance(%)')
plt.title('Feature Importance for Iris')
plt.savefig("D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/decision trees/importance.png")

dot_data = export_graphviz(model,
                           out_file=None,
                           feature_names=X.columns,
                           class_names=model.classes_,
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("iris_tree.png")
Image(graph.create_png())
