

def predict(X_test, model):
    
    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred
