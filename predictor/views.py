import pickle
import os
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))

def home(request):
    if request.method == 'POST':

        data = [
            int(request.POST['creditscore']),
            int(request.POST['age']),
            float(request.POST['balance']),
            int(request.POST['tenure']),
            float(request.POST['salary'])
        ]

        data = scaler.transform([data])

        prediction = model.predict(data)
        prob = model.predict_proba(data)

        probability = round(prob[0][1] * 100, 2)
        retention = round(100 - probability, 2)

        result = "Churn" if prediction[0] == 1 else "Not Churn"

        return render(request, 'result.html', {
            'result': result,
            'probability': probability,
            'retention': retention
        })

    return render(request, 'form.html')