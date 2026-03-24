import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings

# Global dictionary holding models
MODELS_DICT = None

def predict_single_row(row):
    val = row['speed_limit']
    
    # Extract features for equations
    lighting_dim = 1 if row['lighting'] == 'dim' else 0
    lighting_night = 1 if row['lighting'] == 'night' else 0
    weather_foggy = 1 if row['weather'] == 'foggy' else 0
    weather_rainy = 1 if row['weather'] == 'rainy' else 0
    curvature = row['curvature']
    speed_limit = row['speed_limit']
    num_reported_accidents = row['num_reported_accidents']
    
    if pd.isna(val) or val <= 40:
        # Low Case
        result = 0.2898
        + (0.0874) * curvature\
        + (0.0007) * speed_limit\
        + (0.0136) * num_reported_accidents\
        + (0.0007) * lighting_dim\
        + (0.0896) * lighting_night\
        + (0.0443) * weather_foggy\
        + (0.0436) * weather_rainy\
                 
    elif 40 < val <= 60:
        # Medium Case
        result = 0.4516\
        + (0.0883) * curvature\
        - (0.0000) * speed_limit\
        + (0.0155) * num_reported_accidents\
        + (0.0002) * lighting_dim\
        + (0.0806) * lighting_night\
        + (0.0451) * weather_foggy\
        + (0.0441) * weather_rainy\
                 
    elif 60 < val <= 100:
        # High Case
        result = 0.3734\
        + (0.0871) * curvature\
        + (0.0934) * speed_limit\
        + (0.0127) * num_reported_accidents\
        + (0.0006) * lighting_dim\
        + (0.0893) * lighting_night\
        + (0.0454) * weather_foggy\
        + (0.0449) * weather_rainy\
                 
    else: # val > 120
        # Extreme Case
        result = 0.3526\
        + (0.0884) * curvature\
        + (0.0805) * speed_limit\
        + (0.0141) * num_reported_accidents\
        - (0.0001) * lighting_dim\
        + (0.0859) * lighting_night\
        + (0.0438) * weather_foggy\
        + (0.0427) * weather_rainy\
                 
    return result

def index(request):
    result = None
    row = None
    if request.method == 'POST':
        # Ensure we capture numerical errors safely
        try:
            row = {
                'lighting': request.POST.get('lighting'),
                'weather': request.POST.get('weather'),
                'curvature': max(0.0, float(request.POST.get('curvature'))),
                'speed_limit': max(0.0, float(request.POST.get('speed_limit'))),
                'num_reported_accidents': max(0.0, float(request.POST.get('num_reported_accidents'))),
            }
            pred = predict_single_row(row)
            pred_clamped = max(0.0, min(1.0, pred))
            result = f"{pred_clamped * 100:.2f}%"
        except Exception as e:
            result = f"Error interpreting input: {str(e)}"
            
    is_error = result is not None and "Error" in result
    return render(request, 'predictor/index.html', {'result': result, 'row': row, 'is_error': is_error})
