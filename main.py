import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import visualizations

# Загрузка данных
data = pd.read_csv("ecommerce_customer_data_large.csv")

# Удаление строк с пропущенными значениями в целевой переменной
data = data.dropna(subset=['Returns'])

# Подготовка данных
X = data[['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age']]
y = data['Returns']

# Кодирование категориальных признаков, если они есть
# Пример:
# label_encoder = LabelEncoder()
# X['Payment Method'] = label_encoder.fit_transform(X['Payment Method'])

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Вызов функций для создания визуализаций
visualizations.plot_roc_curve(y_test, y_pred)
visualizations.plot_feature_importances(model, X_train.columns)
visualizations.plot_confusion_matrix(y_test, y_pred)