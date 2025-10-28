trend_forecasting = """
Trend Forecasting Module
Predicts future fatigue levels using time-series analysis
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

class TrendForecaster:
    def __init__(self):
        self.model = None
    
    def forecast(self, data_df, days_ahead=7):
        """
        Forecast future fatigue levels
        data_df: DataFrame with 'timestamp' and 'fatigue_score' columns
        days_ahead: number of days to forecast
        """
        try:
            # Prepare time series data
            df = data_df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.resample('D').mean()  # Daily average
            df = df.fillna(method='ffill')
            
            scores = df['fatigue_score'].values
            
            if len(scores) < 3:
                # Use simple linear regression for small datasets
                return self._simple_forecast(scores, days_ahead)
            
            # Use exponential smoothing for larger datasets
            try:
                model = ExponentialSmoothing(
                    scores,
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                )
                fitted_model = model.fit()
                forecast = fitted_model.forecast(days_ahead)
                
                # Generate forecast dates
                last_date = df.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=days_ahead,
                    freq='D'
                )
                
                forecast_data = []
                for date, value in zip(forecast_dates, forecast):
                    forecast_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'predicted_fatigue': float(max(0, min(10, value))),
                        'confidence': 'medium'
                    })
                
                return forecast_data
                
            except:
                return self._simple_forecast(scores, days_ahead)
                
        except Exception as e:
            print(f"Forecasting error: {e}")
            return self._default_forecast(days_ahead)
    
    def _simple_forecast(self, scores, days_ahead):
        """Simple linear regression forecast"""
        X = np.arange(len(scores)).reshape(-1, 1)
        y = scores
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(scores), len(scores) + days_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        forecast_data = []
        for i, pred in enumerate(predictions):
            forecast_data.append({
                'date': f"Day {i+1}",
                'predicted_fatigue': float(max(0, min(10, pred))),
                'confidence': 'low'
            })
        
        return forecast_data
    
    def _default_forecast(self, days_ahead):
        """Return default forecast if all methods fail"""
        return [{
            'date': f"Day {i+1}",
            'predicted_fatigue': 5.0,
            'confidence': 'very_low'
        } for i in range(days_ahead)]