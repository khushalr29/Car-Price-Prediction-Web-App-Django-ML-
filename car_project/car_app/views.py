from django.shortcuts import render
import joblib
import pandas as pd
import os
from django.conf import settings

# Load pipeline (encoding + scaling + model)
model_path = os.path.join(settings.BASE_DIR, "ml", "car_price_pipeline.pkl")

pipeline = joblib.load(model_path)

def home(request):
    context = {}
    if request.method == "POST":
        year = request.POST.get('year')
        price = request.POST.get('present_price')
        kms = request.POST.get('kms')
        owner = request.POST.get('owner')
        fuel = request.POST.get('Fuel_Type')
        seller = request.POST.get('Seller_Type')
        transmission = request.POST.get('Transmission')

        # Build dataframe
        input_data = pd.DataFrame([{
            "Year": int(year),
            "Present_Price": float(price),
            "Kms_Driven": int(kms),
            "Owner": int(owner),
            "Fuel_Type": fuel,
            "Seller_Type": seller,
            "Transmission": transmission
        }])

        # Predict
        prediction = round(pipeline.predict(input_data)[0], 2)


        context = {
            "prediction": prediction,
            "year": year,
            "present_price": price,
            "kms": kms,
            "owner": owner,
            "fuel": fuel,
            "seller": seller,
            "trans": transmission
        }
        return render(request, "result.html", context)
    return render(request, "home.html")
