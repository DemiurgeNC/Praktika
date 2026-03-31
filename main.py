import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append('src')

from data_preprocessing import DataPreprocessor
from effectiveness import EffectivenessCalculator

def main():
    print("=" * 60)
    print("Moneyball Football Analytics System")
    print("Оценка эффективности игроков и анализ «цена/качество»")
    print("=" * 60)

    data_path = "data/raw/football_data.csv"

    if not os.path.exists(data_path):
        print(f"Ошибка: Файл данных не найден: {data_path}")
        print("Пожалуйста, поместите данные в data/raw/football_data.csv")
        return

    print("\n1. Загрузка и обработка данных...")
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.load_data()
    data = preprocessor.clean_data(min_minutes=900)

    print(f"\nРазмер данных после очистки: {data.shape}")
    print(f"Количество игроков: {len(data)}")
    if 'market_value' in data.columns:
        print(f"Диапазон стоимости: {data['market_value'].min():.2f} - {data['market_value'].max():.2f} млн €")

    print("\n2. Расчёт интегральной эффективности...")
    available_metrics = []
    possible_metrics = ['xG', 'xA', 'key_passes', 'shots', 'goals', 'assists']
    for m in possible_metrics:
        if m in data.columns or f"{m}_90" in data.columns:
            available_metrics.append(f"{m}_90" if f"{m}_90" in data.columns else m)

    if not available_metrics:
        print("Ошибка: в данных не найдены атакующие метрики (xG, xA, key_passes и т.д.).")
        return

    print(f"Используемые метрики: {available_metrics}")

    weights = {
        'xG_90': 0.35,
        'xA_90': 0.35,
        'key_passes_90': 0.15,
        'shots_90': 0.15
    }
    weights = {k: v for k, v in weights.items() if k.replace('_90','') in available_metrics or k in available_metrics}

    calculator = EffectivenessCalculator(data, weights=weights)
    data_with_eff = calculator.compute_effectiveness()
    data_with_vfm = calculator.compute_value_for_money()

    print("\n3. Формирование рейтинга игроков...")
    ranking_vfm = calculator.get_ranking(sort_by='value_for_money', ascending=False, top_n=20)
    ranking_eff = calculator.get_ranking(sort_by='effectiveness', ascending=False, top_n=20)

    print("\nТоп-10 игроков по коэффициенту «цена/качество»:")
    print("-" * 80)
    for idx, row in ranking_vfm.head(10).iterrows():
        print(f"{idx+1:2}. {row['player'][:25]:25} | {row['club'][:15]:15} | "
              f"Эффективность: {row['effectiveness']:.3f} | "
              f"Стоимость: {row['market_value']:.1f} млн | "
              f"Цена/качество: {row['value_for_money']:.4f}")

    print("\nТоп-10 игроков по эффективности:")
    print("-" * 80)
    for idx, row in ranking_eff.head(10).iterrows():
        print(f"{idx+1:2}. {row['player'][:25]:25} | {row['club'][:15]:15} | "
              f"Эффективность: {row['effectiveness']:.3f} | "
              f"Стоимость: {row['market_value']:.1f} млн")

    print("\n4. Генерация аналитических выводов...")
    reports = calculator.generate_textual_reports(top_n=10)
    for report in reports:
        print(report)

    print("\n5. Сохранение результатов...")
    os.makedirs("data/processed", exist_ok=True)
    ranking_vfm.to_csv("data/processed/ranking_value_for_money.csv", index=False)
    ranking_eff.to_csv("data/processed/ranking_effectiveness.csv", index=False)
    data_with_vfm.to_csv("data/processed/full_analysis.csv", index=False)

    print("\nРезультаты сохранены в папке data/processed/")

    print("\n6. Построение графиков...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    top10 = ranking_vfm.head(10)
    plt.barh(top10['player'].str[:20] + '...', top10['value_for_money'], color='skyblue')
    plt.xlabel('Коэффициент «цена/качество»')
    plt.title('Топ-10 игроков по цене/качество')

    plt.subplot(1, 2, 2)
    top10_eff = ranking_eff.head(10)
    plt.barh(top10_eff['player'].str[:20] + '...', top10_eff['effectiveness'], color='lightgreen')
    plt.xlabel('Интегральная эффективность')
    plt.title('Топ-10 игроков по эффективности')

    plt.tight_layout()
    plt.savefig("data/processed/rankings.png")
    print("Графики сохранены в data/processed/rankings.png")

    print("\n" + "=" * 60)
    print("Анализ завершен успешно!")
    print("=" * 60)

if __name__ == "__main__":
    main()
