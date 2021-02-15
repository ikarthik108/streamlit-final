import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
#pip install
import streamlit as st 
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly
#pip install
from gtts import gTTS


from PIL import Image

from explainer import interpret_model

#app=Flask(__name__)
#Swagger(app)

sampled_df=pd.read_csv("data/sampled.csv")


sampled_df['Closeness_5'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT5)  
sampled_df['Closeness_4'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT4) 

sampled_df['Closeness_2'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT2) 
sampled_df['Closeness_1'] = (sampled_df.LIMIT_BAL - sampled_df.BILL_AMT1) 

feature_set=['PAY_1', 'PAY_2', 'BILL_AMT1','BILL_AMT2', 'PAY_AMT1','PAY_AMT2',
             'AGE', 'Closeness_1','Closeness_2',  'Closeness_4']



# =============================================================================
# import app1
# import app2
# import streamlit as st
# PAGES = {
#     "App1": app1,
#     "App2": app2
#     }
# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))
# page = PAGES[selection]
# page.app()
# =============================================================================


def model_selection(model):
    
    if model=='Voting Classifier(DT,KNN,LR)':
        name="voting1"+'.pkl'
    elif model=='Voting Classifier(DT,KNN,GNB)':
        name="voting2"+'.pkl'
    elif model=='StackingClassifier(KNN,GNB,LR)':
        name="stacking3"+'.pkl'
                
    elif model=='StackingClassifier(RF,GNB,LR)':
        name="stacking4"+'.pkl'
        
    elif model=='StackingClassifier(KNN,RF,GNB)':
        name="stacking1"+'.pkl'
    elif model=='StackingClassifier(KNN,RF,LR)':
        name="stacking2"+'.pkl'
   
    elif model=='Random Forest':
        name="rf.pkl"
        
    
    pickle_in = open(name,"rb")
    classifier=pickle.load(pickle_in)
    return classifier

#@app.route('/')
def welcome():
    return "Welcome All"

def gauge_chart(default_probability,pot_probability):
 
  value1=default_probability
  value1=value1*100
  
  
  
  fig1 = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value =float(value1),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Default Confidence ", 'font': {'size': 24}},
    delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 25], 'color': 'cyan'},
            {'range': [25, 40], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 40}}))

  fig1.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
  
  value2=pot_probability
  value2=value2*100
  
  fig2 = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value =float(value2),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': " Timely Payment Confidence ", 'font': {'size': 24}},
    delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 25], 'color': 'cyan'},
            {'range': [25, 40], 'color': 'royalblue'}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 40}}))

  fig2.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
  
  
  return fig1,fig2

 
def instance_values(PAY_1,PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2,AGE,Closeness_1,Closeness_2,Closeness_4):
    values=list()
    values.append(PAY_1)
    values.extend((PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2,AGE,Closeness_1,Closeness_2,Closeness_4))
    return values

#@app.route('/predict',methods=["Get"])
def predict_default(PAY_1,PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2,AGE,Closeness_1,Closeness_2,Closeness_4,option):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    classifier=model_selection(option)
    int_features = [x for x in [PAY_1,PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2,AGE,Closeness_1,Closeness_2,Closeness_4]]
    final_features = [np.array(int_features)]
    prediction=classifier.predict(final_features)
    default_probability=classifier.predict_proba(final_features)[:,1]
    default_probability=np.round(default_probability,2)
    pot_probability=classifier.predict_proba(final_features)[:,0]
    pot_probability=np.round(pot_probability,2)
    fig1,fig2=gauge_chart(default_probability,pot_probability)
    print(prediction)
    return prediction, default_probability,fig1,fig2




 
def main():
    st.title("Default Predictor")
    st.markdown("<h1 style='text-align: center; color: red;'>Default Predictor</h1>", unsafe_allow_html=True)
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Card Default Payment Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
  
# =============================================================================
#     PAY_1 = st.text_input("PAY_1","Repayment Status in September")
#     PAY_2 = st.text_input("PAY_2","Repayment Status in August")
#     PAY_3 = st.text_input("PAY_3","Repayment Status in July")
#     PAY_4 = st.text_input("PAY_4","Repayment Status in June")
#     PAY_5 = st.text_input("PAY_5","Repayment Status in May")
#     PAY_6 = st.text_input("PAY_6","Repayment Status in April")
# =============================================================================
        
    NAME = st.text_input("Name of the Client")

    PAY_1 = st.selectbox(
    'PAY_1',
     (0,1,2,3,4,5,6,7,8))
    PAY_2 = st.selectbox(
    'PAY_2',
     (0,1,2,3,4,5,6,7,8))
    

    BILL_AMT1 = st.number_input("BILL_AMT1")
    BILL_AMT1=int(BILL_AMT1)
    
    BILL_AMT2 = st.number_input("BILL_AMT2")
    BILL_AMT2=int(BILL_AMT2)
  
    
    PAY_AMT1 = st.number_input("PAY_AMT1")
    PAY_AMT1=int(PAY_AMT1)

    PAY_AMT2 = st.number_input("PAY_AMT2")
    PAY_AMT2=int(PAY_AMT2)
   

    AGE = st.number_input("AGE")
    AGE=int(AGE)
# =============================================================================
#     Closeness_1 = st.number_input("Closeness_1")
#     Closeness_2 = st.number_input("Closeness_2")
#     #Closeness_3 = st.text_input("Closeness_3","Repayment Status in April")
# 
#     Closeness_4 = st.number_input("Closeness_4")
# =============================================================================
    #Closeness_6 = st.text_input("Closeness_6","Repayment Status in April")
    LIMIT_BAL=st.number_input("LIMIT_BAL")
    LIMIT_BAL=int(LIMIT_BAL)
   
    
    BILL_AMT4=st.number_input("BILL_AMT4")
    BILL_AMT4=int(BILL_AMT4)


    
    
    Closeness_1=(LIMIT_BAL) - (BILL_AMT1)
    Closeness_2=(LIMIT_BAL) - (BILL_AMT2)
    Closeness_4=(LIMIT_BAL) - (BILL_AMT4)


 # top_12_rf=['PAY_1', 'PAY_2', 'BILL_AMT1', 'PAY_AMT1', 'AGE', 
               #'Closeness_1', 'PAY_AMT2', 'Closeness_4', 'BILL_AMT2', 'Closeness_2', 'Closeness_3', 'Closeness_6']   
    
    option = st.selectbox(
    'Select A Classifier',
     ('Voting Classifier(DT,KNN,LR)', 'Voting Classifier(DT,KNN,GNB)',
      'StackingClassifier(KNN,GNB,LR)','StackingClassifier(RF,GNB,LR)',
      'StackingClassifier(KNN,RF,GNB)','StackingClassifier(KNN,RF,LR)','Random Forest'))

    st.write('You selected:', option)
    
    values=instance_values(PAY_1,PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2,AGE,Closeness_1,Closeness_2,Closeness_4)
    
    print(type(values))
    
    
    result,probability,text="","",""
    if st.button("Predict"):
        with st.spinner('Predicting The Results...'):
            
            result, probability,fig1,fig2 = predict_default(PAY_1,PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2,
                                                            AGE,Closeness_1,Closeness_2,Closeness_4,
                                                            option)
            if result==1:
                name=NAME
                text="{} is more likely to default next month payment.".format(name)
                st.success('{}.  \nProbability of default is {}  '.format(text,probability))
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                
                #Using Google Text To Speech API
                ta_tts = gTTS('{}. Probability of default is {} percent, which is {} percent more than the threshold value of 50%. '.format(text,
                                                                                                                                            probability*100,(probability*100)-50))
                ta_tts.save("trans.mp3")
                audio_file = open("trans.mp3","rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/ogg")
# =============================================================================
#                 audio_file = open('default.wav', 'rb')
#                 audio_bytes = audio_file.read()
#                 st.audio(audio_bytes, format='audio/wav')
# =============================================================================
            else:
                name=NAME
                text=" {} is likely to Make the Payment on Time".format(name)
                st.success('{}.  \n Probability of default is {}  '.format(text,probability))
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)
                ta_tts = gTTS('{}. probability of default is {} percent which is {} percent less than the threshold value of 50% '.format(text,probability*100,50-(probability*100)))
                ta_tts.save("trans.mp3")
                audio_file = open("trans.mp3","rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/ogg")
    if st.button("Explain Results"):
        
        #Lime Explainer
        classifier=model_selection(option)
        limeexplainer=interpret_model(sampled_df,feature_set,classifier)
        values=instance_values(PAY_1,PAY_2,BILL_AMT1,BILL_AMT2,PAY_AMT1,PAY_AMT2, AGE ,Closeness_1,Closeness_2,Closeness_4)
        explainable_exp = limeexplainer.explain_instance(np.array(values), classifier.predict_proba, num_features=len(feature_set))
        explain = explainable_exp.as_pyplot_figure()
        st.pyplot(explain)
        # Display explainer HTML object
        components.html(explainable_exp.as_html(), height=800)
    
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
# =============================================================================
# file_to_be_uploaded = st.file_uploader("Choose an audio...", type="wav")
# =============================================================================
