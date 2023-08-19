from django.shortcuts import render
# Create your views here.
# from pathlib import Path
# file_here = Path(__file__).parent
from pickle import load
model_file_01 = "D:/Tash_ML_Project/cyberbully/model_and_vectorizer/logisticRegModel.model"
model_nb_file = "D:/Tash_ML_Project/cyberbully/model_and_vectorizer/Model_NB.pk"
vect_filepath = "D:/Tash_ML_Project/cyberbully/model_and_vectorizer/vectorizer.pk"
vectorizer = load(open(file=vect_filepath, mode='rb'))
model_01 = load(open(file=model_file_01, mode='rb'))
model_nb = load(open(file=model_nb_file, mode='rb'))

def index(request):
    if request.method == 'POST':
        input_text = request.POST.get('message').lower()
        choose_model = request.POST.get('choose')
        # print(choose_model)
        if choose_model == "LogisticRegression": 
            print("Using LogisticRegression model")
            vect_input_text = vectorizer.transform([input_text])
            make_prediction = model_01.predict(vect_input_text)
            if make_prediction == [0]:
                prediction = f"This '{input_text}' is abusive 🚩😡"
            else:
                prediction = f"The text '{input_text}' is not threatening"
            return render(request, 'myapp/index.html', {'prediction': prediction})    
        elif choose_model == "naiveBayes":
            print("Using naive Bayes")
            vect_input_text = vectorizer.transform([input_text])
            make_prediction = model_01.predict(vect_input_text.toarray())
            if make_prediction == [0]:
                prediction = f"The text '{input_text}' is abusive 🚩😡"
            else:
                prediction = f"The text '{input_text}' is not threatening"
            return render(request, 'myapp/index.html', {'prediction': prediction})
        else:
            prediction = f'you have not selected a model'
            return render(request, 'myapp/index.html', {'prediction': prediction})
    return render(request, 'myapp/index.html')
