def recommend_models(task):

    recommendations = {
        'classification': [
            'Logistic Regression',
            'Random Forest',
            'XGBoost',
            'SVM'
        ],

        'regression': [
            'Linear Regression',
            'Random Forest Regressor',
            'XGBoost Regressor'
        ],

        'clustering': [
            'KMeans',
            'DBSCAN',
            'Hierarchical Clustering'
        ],

        'time_series': [
            'ARIMA',
            'Prophet',
            'LSTM'
        ]
    }
    return recommendations.get(task);