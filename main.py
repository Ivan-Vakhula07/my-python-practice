import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# 1. ВХІДНІ ДАНІ (Варіант №7)
# ==========================================
intervals = [
    (450, 460),
    (460, 470),
    (470, 480),
    (480, 490),
    (490, 500),
    (500, 510),
    (510, 520),
    (520, 530),
]
frequencies = np.array([2, 7, 14, 25, 24, 20, 5, 2])
n = np.sum(frequencies)  # Обсяг вибірки n = 99
gamma = 0.90  # Задана надійність γ = 0.90

# Середини інтервалів (x_i)
x_i = np.array([(a + b) / 2 for a, b in intervals])

# ==========================================
# 2. РОЗРАХУНКОВА ТАБЛИЦЯ ТА ТОЧКОВІ ОЦІНКИ
# ==========================================
ni_xi = frequencies * x_i
sum_ni_xi = np.sum(ni_xi)

# Вибіркове середнє (x_mean)
x_mean = sum_ni_xi / n

# Відхилення та їх квадрати
diff_x = x_i - x_mean
diff_x_sq = diff_x**2
ni_diff_x_sq = frequencies * diff_x_sq
sum_ni_diff_x_sq = np.sum(ni_diff_x_sq)

# Дисперсії та СКВ
d_var = sum_ni_diff_x_sq / n  # Зміщена дисперсія
s_sq = (n / (n - 1)) * d_var  # Виправлена дисперсія s^2
s = np.sqrt(s_sq)  # Виправлене СКВ s

# ==========================================
# 3. ІНТЕРВАЛЬНІ ОЦІНКИ (γ = 0.90)
# ==========================================
k = n - 1  # Степені вільності k = 98
alpha = 1 - gamma

# t-квантиль Стьюдента
t_gamma = stats.t.ppf(1 - alpha / 2, df=k)
se_mean = s / np.sqrt(n)
delta = t_gamma * se_mean

# Довірчий інтервал для m
m_min = x_mean - delta
m_max = x_mean + delta
epsilon = (delta / x_mean) * 100

# Критичні точки хі-квадрат
chi2_1 = stats.chi2.ppf(1 - alpha / 2, df=k)  # Права (більша)
chi2_2 = stats.chi2.ppf(alpha / 2, df=k)  # Ліва (менша)

# Довірчий інтервал для σ^2
var_min = (k * s_sq) / chi2_1
var_max = (k * s_sq) / chi2_2

# Довірчий інтервал для σ
sigma_min = np.sqrt(var_min)
sigma_max = np.sqrt(var_max)

# ==========================================
# 4. ВИВЕДЕННЯ РЕЗУЛЬТАТІВУ КОНСОЛЬ
# ==========================================
print("=" * 70)
print(f"ТАБЛИЦЯ 1.2 — Розрахункова таблиця (n = {n})")
print("=" * 70)
print(
    f"{'i':<3} | {'Інтервали':<10} | {'x_i':<5} | {'n_i':<4} | {'n_i*x_i':<8} | {'x_i - x_mean':<12} | {'n_i*(x_i - x_mean)^2':<20}"
)
print("-" * 70)
for i in range(len(intervals)):
    int_str = f"{intervals[i][0]}-{intervals[i][1]}"
    print(
        f"{i+1:<3} | {int_str:<10} | {x_i[i]:<5.1f} | {frequencies[i]:<4} | {ni_xi[i]:<8.1f} | {diff_x[i]:<12.4f} | {ni_diff_x_sq[i]:<20.4f}"
    )
print("-" * 70)
print(
    f"Сума: {'':<16} | {n:<4} | {sum_ni_xi:<8.1f} | {'-':<12} | {sum_ni_diff_x_sq:<20.4f}"
)
print("=" * 70)

print("\nТОЧКОВІ ОЦІНКИ:")
print(f"Вибіркове середнє (x_mean) = {x_mean:.4f} кПа")
print(f"Зміщена дисперсія (d)      = {d_var:.4f} кПа^2")
print(f"Виправлена дисперсія (s^2)  = {s_sq:.4f} кПа^2")
print(f"Виправлене СКВ (s)         = {s:.4f} кПа")

print(f"\nІНТЕРВАЛЬНІ ОЦІНКИ (γ = {gamma}):")
print(f"t-квантиль t_gamma         = {t_gamma:.4f}")
print(f"Точність оцінки (delta)    = {delta:.4f} кПа")
print(f"Довірчий інтервал для m    = ({m_min:.4f}; {m_max:.4f}) кПа")
print(f"Відносна похибка (epsilon) = {epsilon:.2f}%")
print(
    f"Критичні точки χ^2         = χ1^2 = {chi2_1:.3f}, χ2^2 = {chi2_2:.3f}"
)
print(f"Довірчий інтервал для σ^2  = ({var_min:.4f}; {var_max:.4f}) кПа^2")
print(f"Довірчий інтервал для σ    = ({sigma_min:.4f}; {sigma_max:.4f}) кПа")

# ==========================================
# 5. ПОБУДОВА ГРАФІКА (Гістограма + Полігон)
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

# Гістограма
interval_borders = [a for a, b in intervals] + [intervals[-1][1]]
ax.bar(
    x_i,
    frequencies,
    width=10,
    color="#9ecae1",
    edgecolor="#1f4e79",
    linewidth=1.2,
    align="center",
)

# Полігон частот
ax.plot(
    x_i,
    frequencies,
    color="#b85420",
    marker="o",
    linewidth=2,
    markersize=6,
    label="полігон частот",
)

# Значення частот над точками
for x, y in zip(x_i, frequencies):
    ax.annotate(
        f"{y}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
        fontsize=10,
    )

# Лінія вибіркового середнього
ax.axvline(
    x=x_mean,
    color="#333333",
    linestyle="--",
    linewidth=1.5,
    label=f"вибіркове середнє = {x_mean:.2f}",
)

# Налаштування осей та сітки
ax.set_xlabel("тиск у гидросистемі, кПа", fontsize=11)
ax.set_ylabel("частота $n_i$", fontsize=11)
ax.set_xticks(interval_borders)
ax.set_ylim(0, 30)
ax.grid(axis="y", color="#e0e0e0", linestyle="-", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=10)

plt.title(
    f"Рисунок 1.1 — Гістограма та полігон частот ($n = {n}$)",
    y=-0.22,
    fontsize=12,
)
plt.tight_layout()

# Збереження та відображення
plt.savefig("histogram_v7.png", dpi=300)
plt.show()