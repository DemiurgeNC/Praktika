import pandas as pd
import numpy as np

class EffectivenessCalculator:
    def __init__(self, data, weights=None):
        self.data = data
        self.weights = weights or {
            'xG_90': 0.35,
            'xA_90': 0.35,
            'key_passes_90': 0.15,
            'shots_90': 0.15
        }
        self.effectiveness = None

    def add_per_90_metrics(self, metric_cols):
        for col in metric_cols:
            if col not in self.data.columns and col.replace('_90', '') in self.data.columns:
                base = col.replace('_90', '')
                self.data[col] = self.data[base] / self.data['minutes'] * 90
        return self.data

    def compute_effectiveness(self, metric_cols=None):
        if metric_cols is None:
            metric_cols = list(self.weights.keys())

        self.add_per_90_metrics(metric_cols)

        effective_data = self.data.copy()
        for col in metric_cols:
            if col not in effective_data.columns:
                raise ValueError(f"Метрика {col} отсутствует в данных.")
            mean = effective_data[col].mean()
            std = effective_data[col].std()
            if std == 0:
                effective_data[col + '_norm'] = 0
            else:
                effective_data[col + '_norm'] = (effective_data[col] - mean) / std

        self.effectiveness = np.zeros(len(effective_data))
        for col in metric_cols:
            self.effectiveness += self.weights[col] * effective_data[col + '_norm']

        self.data['effectiveness'] = self.effectiveness
        return self.data

    def compute_value_for_money(self):
        if 'effectiveness' not in self.data.columns:
            raise ValueError("Сначала вызовите compute_effectiveness()")
        self.data['value_for_money'] = self.data['effectiveness'] / self.data['market_value']
        return self.data

    def get_ranking(self, sort_by='value_for_money', ascending=False, top_n=None):
        if sort_by not in self.data.columns:
            raise ValueError(f"Колонка {sort_by} не найдена.")
        ranking = self.data.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        if top_n:
            ranking = ranking.head(top_n)
        return ranking

    def generate_textual_reports(self, top_n=10):
        if 'value_for_money' not in self.data.columns:
            self.compute_value_for_money()

        top_players = self.get_ranking(sort_by='value_for_money', ascending=False, top_n=top_n)

        reports = []
        for idx, row in top_players.iterrows():
            name = row['player']
            club = row['club']
            eff = row['effectiveness']
            vfm = row['value_for_money']
            value = row['market_value']
            age_info = f", возраст {row['age']} лет" if 'age' in row else ""

            if vfm > 0.5:
                assessment = "является очень выгодным приобретением"
            elif vfm > 0.2:
                assessment = "показывает хорошее соотношение цена/качество"
            else:
                assessment = "стоит дороже, чем его текущая эффективность"

            report = (f"Игрок {name} ({club}){age_info}. "
                      f"Интегральная эффективность: {eff:.3f}, "
                      f"рыночная стоимость: {value:.2f} млн €, "
                      f"коэффициент «цена/качество»: {vfm:.4f}. "
                      f"Вывод: {assessment}.")
            reports.append(report)
        return reports
