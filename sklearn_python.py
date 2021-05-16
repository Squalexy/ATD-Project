import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def main():
	csv = pd.read_csv("features.csv")
	csv = csv.dropna(axis=1)  # apga as linhas com valores em falta (Nan) ; axis = 1 é a coluna
	y = list(csv['activity'])

	# mudar valores de [] consoante o tipo de atividade que queremos analisar
	y = [0 if x in [4, 5, 6] else 1 for x in y]  # atividades estáticas, 0 se for estática, 1 se n for

	X = csv.drop('activity', axis=1)
	print(list(X.columns))

	# ---------------- ALGORITMO SELECIONA AS MELHORES FEATURES ---------------- #

	"""
	selector = SelectKBest(f_classif, k=10)  # estimador da qualidade das features, escolhe as 3 melhores

	selector = selector.fit(X, y)
	# fit faz treino desse estimador, transform transforma matriz e apaga features que n interessam

	X_new = selector.transform(X)
	mask = selector.get_support()
	X = pd.DataFrame(X_new, columns=[f for b, f in zip(mask, X.columns) if b])
	print(X.columns)"""

	# ---------------- ALGORITMO SELECIONA AS FEATURES QUE QUISER ---------------- #

	X = X[['0_Absolute energy', '0_Area under the curve', '0_Autocorrelation']]
	print(X)

	# criar o modelo
	model = DecisionTreeClassifier()

	# pega em x e y ( 2 arrays ), parte. 90% vão para train, 10% para test
	# num faz a aproximacao, no outro verifica-se
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

	# treinar o modelo com os pontos que tinha
	model.fit(X_train, y_train)

	# ver quao bom o modelo é
	print(model.score(X_test, y_test))
	plot_tree(model, feature_names=X.columns)
	plt.show()

	# compara as previsões com o que estaria certo e cria relatório sobre a qualidade do modelo
	y_pred = model.predict(X_test, y_test)
	print(classification_report(y_test, y_pred))


if __name__ == '__main__':
	main()
