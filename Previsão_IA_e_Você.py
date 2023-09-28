import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dt = pd.read_csv("C:/Users/Cliente/Downloads/dfdengue.csv")
dt = dt.dropna()
x = dt.iloc[:, :-28]
y = dt.iloc[:, -28]

# Separando dado de teste com treino
X_train, X_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=0)

# Remover o ano de 2030 dos dados de treinamento
train_indices = X_train[X_train.iloc[:, 0] != 2030].index
X_train_filtered = X_train.loc[train_indices]
y_train_filtered = y_train.loc[train_indices]

# Criar uma instância do MinMaxScaler
scaler = MinMaxScaler()

# Normalizar os dados de treino e teste
X_train_normalized = scaler.fit_transform(X_train_filtered)
X_test_normalized = scaler.transform(X_test)

# Treinar o modelo com os dados normalizados
regressor = lm.LinearRegression()
regressor.fit(X_train_normalized, y_train_filtered)

# Realizar previsões para os dados de treino e teste
y_train_predicted = regressor.predict(X_train_normalized)
y_test_predicted = regressor.predict(X_test_normalized)

# Desnormalizar as previsões para voltar à escala original
y_train_predicted = scaler.inverse_transform(y_train_predicted.reshape(-1, 1))
y_test_predicted = scaler.inverse_transform(y_test_predicted.reshape(-1, 1))

# Plotar o gráfico com os dados de treino filtrados (treino)
plt.scatter(X_train_filtered.iloc[:, 0], y_train_filtered, color='red', label='Dados de Treino Filtrados')
plt.title('Dados de Treino Filtrados')
plt.legend()
plt.show()

# Plotar o gráfico com as previsões para os dados de treino filtrados (treino)
plt.scatter(X_train_filtered.iloc[:, 0], y_train_predicted, color='blue', label='Previsões (Treino Filtrado)')
plt.title('Previsões para Dados de Treino Filtrados')
plt.legend()
plt.show()

# Plotar o gráfico com os dados de teste (teste)
plt.scatter(X_test.iloc[:, 0], y_test, color='red', label='Dados de Teste')
plt.title('Dados de Teste')
plt.legend()
plt.show()

# Plotar o gráfico com as previsões para os dados de teste (teste)
plt.scatter(X_test.iloc[:, 0], y_test_predicted, color='blue', label='Previsões (Teste)')
plt.title('Previsões para Dados de Teste')
plt.legend()
plt.show()
