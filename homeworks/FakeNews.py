import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import nltk

# Загрузка NLTK данных
nltk.download('stopwords')


class TextTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для текстовых признаков"""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        if not isinstance(text, str):
            return ''

        # Удаление URL и email
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)

        # Базовая очистка
        text = text.lower()
        text = re.sub(r'[^\w\s]|_', '', text)
        text = re.sub(r'\d+', '', text)

        # Стемминг
        words = [self.stemmer.stem(word) for word in text.split()
                 if len(word) > 2 and word not in self.stop_words]
        return ' '.join(words)

    def transform(self, X):
        return X['text'].apply(self.clean_text)

    def fit(self, X, y=None):
        return self


class NumericTransformer(BaseEstimator, TransformerMixin):
    """Трансформер для числовых признаков"""

    def transform(self, X):
        # Создаем числовые признаки
        features = pd.DataFrame()
        features['text_len'] = X['text'].apply(lambda x: len(str(x)))
        features['capitals'] = X['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
        features['exclamations'] = X['text'].apply(lambda x: str(x).count('!'))
        features['has_quotes'] = X['text'].apply(lambda x: 1 if '"' in str(x) else 0)
        return features

    def fit(self, X, y=None):
        return self


def main():
    # Загрузка данных
    train = pd.read_csv('train_fn.csv')
    test = pd.read_csv('test_fn.csv')

    # Инициализация компонентов
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8,
        stop_words='english',
        sublinear_tf=True
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42,
        max_features='sqrt'
    )

    # Создание пайплайна
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('text_clean', TextTransformer()),
                ('tfidf', vectorizer)
            ])),
            ('numeric', Pipeline([
                ('num_features', NumericTransformer()),
                ('scaler', StandardScaler())
            ]))
        ])),
        ('clf', model)
    ])

    # Обучение модели
    print("Training model...")
    pipeline.fit(train, train['label'])

    # Кросс-валидация
    scores = cross_val_score(pipeline, train, train['label'], cv=3, scoring='f1')
    print(f"Cross-validation F1: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Калибровка (опционально)
    calibrated_model = CalibratedClassifierCV(pipeline, cv=3, method='isotonic')
    calibrated_model.fit(train, train['label'])

    # Предсказание
    print("Making predictions...")
    test_pred = calibrated_model.predict(test)

    # Сохранение результатов
    submission = pd.DataFrame({'id': test['id'], 'label': test_pred})
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    main()