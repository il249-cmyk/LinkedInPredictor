#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
s = pd.read_csv("social_media_usage.csv")


# In[11]:


import numpy as np
def clean_sm(input):
    return np.where(input==1,1,0)


# In[10]:


toys={'toys':['bear','fox','elk'],'price':[12,1,1]}
toydataframe=pd.DataFrame(toys)


# In[14]:


clean_sm(toydataframe)


# In[53]:


cleans=clean_sm(s)
column_number=s.columns.get_loc("web1h")
column_number_marital=s.columns.get_loc("marital")
column_number_parent=s.columns.get_loc("par")
Married=cleans[:,column_number_marital]
Parent=cleans[:,column_number_parent]
LinkedIn=cleans[:,column_number]
ss=pd.DataFrame({
    'sm_li':LinkedIn,
    'Income':s.income,
    'Education':s.educ2,
    'Parent':Parent,
    'Married':Married,
    'Female':s.gender,
    'Age':s.age
})
ss.Female=np.where(ss.Female==2,1,0)
ss.Income=np.where(ss.Income>8,pd.NA,ss.Income)
ss.Education=np.where(ss.Education>8,pd.NA,ss.Education)
ss.Age=np.where(ss.Age>98,pd.NA,ss.Age)
cleanss=ss.dropna()


# In[54]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(cleanss)
plt.suptitle('Pair Plot of Variables', y=1.02)
plt.tight_layout()
#commenting out so that the results print smoother in part 2
#plt.show()


# In[57]:


#cleanss.describe()


# In[106]:


import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
FeatureSet=cleanss.drop(columns=['sm_li'])
TargetVector=cleanss.sm_li


# In[107]:


X_train, X_test,Y_train,y_test = train_test_split(
    FeatureSet, TargetVector,
    test_size=0.2,
    random_state=150
)


# In[118]:


model=LogisticRegression(class_weight='balanced',random_state=15)
model.fit(X_train,Y_train)
prediction=model.predict(X_test)
accuracy = skl.metrics.accuracy_score(prediction,y_test)

#commenting out so that the results print smoother in part 2
#print(accuracy)


# In[136]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test,prediction)
cm_df=pd.DataFrame(
    cm,
    index=['Actual Positive','Actual Negative'],
    columns=['Predicted Positive','Predicted Negative']
)

#commenting out so that the results print smoother   
#print(cm_df)


# In[137]:


TP=cm_df.iloc[0,0]
FP=cm_df.iloc[1,0]
FN=cm_df.iloc[0,1]
TN=cm_df.iloc[1,1]
Precision = round(TP/(TP+FP),2)
Recall = round(TP/(TP+FN),2)
F1score= round((2*Precision*Recall)/(Precision+Recall),2)


# In[135]:


#commenting out so that the results of part 2 run smoother
#print(f'the precision is {Precision}, the recall is {Recall} and the F1score is {F1score}')
#print(classification_report(y_test,prediction))


# In[148]:


YoungTest = pd.DataFrame({
    'Income':[8],
    'Education':[7],
    'Parent':[0],
    'Married':[1],
    'Female':[1],
    'Age':[42]
})

OldTest = pd.DataFrame({
    'Income':[8],
    'Education':[7],
    'Parent':[0],
    'Married':[1],
    'Female':[1],
    'Age':[82]
})

#commenting out the results of part 1 so that part 2 runs smoother
#print(model.predict(YoungTest))
#print(model.predict_proba(YoungTest))
#print(model.predict(OldTest))
#print(model.predict_proba(OldTest))


# In[170]:


def IncomeHelper(IncomeInput):
    if IncomeInput == "Less than $10,000":
        return 1
    elif IncomeInput == "10 to under $20,000":
        return 2
    elif IncomeInput == "20 to under $30,000":
        return 3
    elif IncomeInput == "30 to under $40,000":
        return 4
    elif IncomeInput == "40 to under $50,000":
        return 5
    elif IncomeInput == "50 to under $75,000":
        return 6
    elif IncomeInput == "75 to under $100,000":
        return 7
    else:
        return 8

def EducationHelper(EducationInput):
    if EducationInput == "Less than high school (Grades 1-8 or no formal schooling)":
        return 1
    elif EducationInput == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
        return 2
    elif EducationInput == "High school graduate (Grade 12 with diploma or GED certificate)":
        return 3
    elif EducationInput == "Some college, no degree (includes some community college)":
        return 4
    elif EducationInput == "Two-year associate degree from a college or university":
        return 5
    elif EducationInput == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
        return 6
    elif EducationInput == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
        return 7
    else:
        return 8

def ParentHelper(input):
    if input == "Have kids":
        return 1
    else:
        return 0

def MarriedHelper(input):
    if input == "Married":
        return 1
    else:
        return 0

def FemaleHelper(input):
    if input == "Yes":
        return 1
    else:
        return 0


# In[178]:


def predictionResults(test):
    if model.predict(test)[0] == True:
        return "You are a likely LinkedIn User"
    else:
        return "You are likely not a LinkedIn User"
modelinput = predictionResults(YoungTest)
output = pd.DataFrame({"Are you a LinkedIn User?", modelinput})
print(output)


# In[214]:


import streamlit as st
import altair as alt
st.title("LinkedIn Predictor")

userIncome =st.selectbox("Enter your Annual Income",["Less than $10,000","10 to under $20,000","20 to under $30,000","30 to under $40,000","40 to under $50,000","50 to under $75,000","75 to under $100,000","100 to under $150,000","$150,000+"])
userIncome = IncomeHelper(userIncome)
userEducation =st.selectbox("Ender your level of Education",["Less than high school (Grades 1-8 or no formal schooling)","High school incomplete (Grades 9-11 or Grade 12 with NO diploma)","High school graduate (Grade 12 with diploma or GED certificate)","Some college, no degree (includes some community college)","Two-year associate degree from a college or university","Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)","Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)","Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])
userEducation = EducationHelper(userEducation)
userParent = st.selectbox("Are you a parent of a child under 18 living in your home?",["Have kids","Do not have kids"])
userParent = ParentHelper(userParent)
userMarried = st.radio("What is your marital status",["Married","Not Married"])
userMarried = MarriedHelper(userMarried)
userFemale = st.radio("Are you female?",["Yes","No"])
userFemale = FemaleHelper(userFemale)
userAge = st.slider("Enter your age",0,97,97)

NewTest = pd.DataFrame({
    'Income': [userIncome],
    'Education':[userEducation],
    'Parent': [userParent],
    'Married':[userMarried],
    'Female':[userFemale],
    'Age':[userAge],
})

Confirmation = st.checkbox("Is the above information correct?")
arr = model.predict_proba(NewTest)
unlikelyChance=float(arr[0][0])
likelyChance=float(arr[0][1])
output = pd.DataFrame({"User Likelihood": ["Not a user","user"], "Percentage": [unlikelyChance,likelyChance]})
chart = alt.Chart(output).mark_bar().encode(
    x=alt.X("User Likelihood", title="Chance of LinkedIn Usership",axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Percentage",axis=alt.Axis(format="%"))
)
if Confirmation:
    st.write("Here are your results:")
    if model.predict(NewTest)[0] == 1:
        st.write("You are a likely LinkedIn User")
    else:
        st.write("you are likely not a LinkedInUser")
    st.altair_chart(chart, use_container_width=True)


# In[ ]:




