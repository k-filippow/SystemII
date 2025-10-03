import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
mat = loadmat("female_2.mat")
print("Ключи файла:", mat.keys())

# Движения
movements = ["cyl", "hook", "tip", "palm", "spher", "lat"]
X_list = []
y_list = []

# Собираем все движения
for label, move in enumerate(movements, start=0):
    ch1 = mat[f"{move}_ch1"]
    ch2 = mat[f"{move}_ch2"]

    # Соединяем каналы
    X_move = np.hstack([ch1, ch2])
    y_move = np.full((X_move.shape[0],), label)  # метка класса

    X_list.append(X_move)
    y_list.append(y_move)

# Общие массивы
X = np.vstack(X_list)
y = np.concatenate(y_list)

print("Форма X:", X.shape)
print("Форма y:", y.shape)
print("Классы:", np.unique(y))

# Разделение выборки
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Масштабирование
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Персептрон
perceptron = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
perceptron.fit(X_train, y_train)
y_pred_perc = perceptron.predict(X_test)
acc_perc = accuracy_score(y_test, y_pred_perc)

# Многослойный персептрон
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=500,
    learning_rate_init=0.001,
    solver="adam",
    random_state=42
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

# Эксперименты
learning_rates = [0.0001, 0.001, 0.01, 0.1]
acc_lr = []
for lr in learning_rates:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                        learning_rate_init=lr, solver="adam", random_state=42)
    mlp.fit(X_train, y_train)
    acc_lr.append(accuracy_score(y_val, mlp.predict(X_val)))

best_lr = learning_rates[np.argmax(acc_lr)]
best_acc_lr = max(acc_lr)

plt.plot(learning_rates, acc_lr, marker="o")
plt.xscale("log")
plt.xlabel("Скорость обучения (learning rate)")
plt.ylabel("Точность на валидации")
plt.title("Влияние learning_rate на точность MLP")
plt.grid(True)
plt.show()

# Alpha (регуляризация)
alphas = [0.0001, 0.001, 0.01, 0.1, 1]
acc_alpha = []
for a in alphas:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                        learning_rate_init=0.001, solver="adam",
                        alpha=a, random_state=42)
    mlp.fit(X_train, y_train)
    acc_alpha.append(accuracy_score(y_val, mlp.predict(X_val)))

best_alpha = alphas[np.argmax(acc_alpha)]
best_acc_alpha = max(acc_alpha)

plt.plot(alphas, acc_alpha, marker="o")
plt.xscale("log")
plt.xlabel("Alpha (регуляризация)")
plt.ylabel("Точность на валидации")
plt.title("Влияние alpha на точность MLP")
plt.grid(True)
plt.show()

# Сравнение оптимизаторов
solvers = ["sgd", "adam", "lbfgs"]
acc_solvers = []
for solver in solvers:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                        learning_rate_init=0.001, solver=solver, random_state=42)
    mlp.fit(X_train, y_train)
    acc_solvers.append(accuracy_score(y_val, mlp.predict(X_val)))

best_solver = solvers[np.argmax(acc_solvers)]
best_acc_solver = max(acc_solvers)

plt.bar(solvers, acc_solvers)
plt.xlabel("Оптимизатор")
plt.ylabel("Точность на валидации")
plt.title("Сравнение оптимизаторов MLP")
plt.show()

print("\nРезультаты")
print(f"Лучший персептрон (тест): {acc_perc:.4f}")
print(f"Лучший MLP (тест): {acc_mlp:.4f}")
print(f"Лучший learning rate: {best_lr} (точность {best_acc_lr:.4f})")
print(f"Лучший alpha: {best_alpha} (точность {best_acc_alpha:.4f})")
print(f"Лучший оптимизатор: {best_solver} (точность {best_acc_solver:.4f})")

