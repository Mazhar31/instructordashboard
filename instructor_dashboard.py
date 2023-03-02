import pickle
from statistics import mean
import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets
import sklearn.metrics


prediction_model = pickle.load(open('aml_model.pkl', 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Instructor Dashboard',                          
                          ['Student Performance Tracking',
                           'Student Engagement',
                           'At-Risk Students',
                           'Manual Prediction',
                           'Course Recommendations'],
                          icons=['activity-fill','cursor-fill','exclamation-octagon-fill','person-check-fill', 'journals'],
                          default_index=0)
    

    course_stage = st.select_slider('Select Stage of Course Completion',
    options=['10', '20', '30', '40', '50', '60', '70','80','90','100'])

df = pd.read_csv('final_merged_'+course_stage+'.csv')
df.loc[df.final_result =='Distinction','final_result'] = 'Pass'
df.loc[df.final_result =='Withdrawn','final_result'] = 'Fail'
df['id_student'] = df['id_student'].astype('object')



#We consider latest semester 2014J as current semester
#And run simulation for 10 to 100 percent course completion for predictions and acitivity/assesment visualizations

#Student Performance Page
if (selected == 'Student Performance Tracking'):
    
    # page title
    st.title('Student Performance Tracking')
    df_latest_semester = df[df['code_presentation'] == '2014J']


    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        check_course_filter = st.checkbox("Filter on Course",value = False,key=1)
    with col2:
        filter_course = st.selectbox('Select Course', df_latest_semester['code_module'].unique())    
    with col3:
        check_student_filter = st.checkbox("Filter on Student",value = False,key=1)
    with col4:
        if check_course_filter:
            df_latest_semester = df_latest_semester[df_latest_semester['code_module'] == filter_course]
        else:
            df_latest_semester = df[df['code_presentation'] == '2014J'] 


        filter_student = st.selectbox('Filter on Student', df_latest_semester['id_student'])

    df_latest_semester1 = df_latest_semester[['id_student', 'code_module' ,'score_count',
       'score_mean', 'score_sum', 'submission_delay_mean',
       'submission_delay_sum', 'weighted_score_mean', 'weighted_score_sum']]    

    if check_course_filter:
        if check_student_filter:
            chart_data = df_latest_semester1[(df_latest_semester1['code_module'] == filter_course) & (df_latest_semester1['id_student'] == filter_student)]
            chart_data = chart_data.append(df_latest_semester1.agg(['mean'])) 
            chart_data.rename(index={'mean':0},inplace=True)
            chart_data.iloc[[1],[0]] = 0
            chart_data.iloc[[1],[1]] = 'Average'
            st.dataframe(chart_data)
            chart_data1 = chart_data.groupby('id_student').first()
            chart_data1.index.name = 'id_student'
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.bar_chart(chart_data1[['score_count']])            
            with col2:
                st.bar_chart(chart_data1[['score_mean']]) 
            with col3:
                st.bar_chart(chart_data1[['submission_delay_mean']])    
            with col4:
                st.bar_chart(chart_data1[['weighted_score_mean' ]])
        else:
            chart_data = df_latest_semester1[df_latest_semester1['code_module'] == filter_course]
            st.dataframe(chart_data,height=400)
            chart_data1 = chart_data.groupby('id_student').first().head(50)
            chart_data1.index.name = 'id_student'
                
            st.bar_chart(chart_data1[['score_count']])            
            st.bar_chart(chart_data1[['score_mean']]) 
            st.bar_chart(chart_data1[['submission_delay_mean']])    
            st.bar_chart(chart_data1[['weighted_score_mean' ]])


            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="score_count")
                st.pyplot(fig)            
            
            with col2:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="score_mean")
                st.pyplot(fig) 


            with col1:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="submission_delay_mean")
                st.pyplot(fig)            
            
            with col2:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="weighted_score_mean")
                st.pyplot(fig) 


    elif check_student_filter:
        st.dataframe(df_latest_semester1[ df_latest_semester1['id_student'] == filter_student])
    else:
        st.dataframe(df_latest_semester1,height=400)

        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="score_count", bins=10)
            st.pyplot(fig)            
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="score_count", bins=10)
            st.pyplot(fig)              



        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="submission_delay_mean",bins=20)
            st.pyplot(fig)            
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="submission_delay_sum", bins=20)
            st.pyplot(fig)       
  

        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="score_mean")
            st.pyplot(fig)           
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="score_sum")
            st.pyplot(fig) 


        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="weighted_score_mean")
            st.pyplot(fig)             

        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="weighted_score_sum")
            st.pyplot(fig) 

# Student Engagement Page
if (selected == 'Student Engagement'):
    
    # page title
    st.title('Student Engagement')
    df_latest_semester = df[df['code_presentation'] == '2014J']

     # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        check_course_filter = st.checkbox("Filter on Course",value = False,key=1)
    with col2:
        filter_course = st.selectbox('Select Course', df_latest_semester['code_module'].unique())    
    with col3:
        check_student_filter = st.checkbox("Filter on Student",value = False,key=1)
    with col4:
        if check_course_filter:
            df_latest_semester = df_latest_semester[df_latest_semester['code_module'] == filter_course]
        else:
            df_latest_semester = df[df['code_presentation'] == '2014J'] 


        filter_student = st.selectbox('Filter on Student', df_latest_semester['id_student'])

    df_latest_semester1 = df_latest_semester[['id_student', 'code_module' ,'mean_forumng', 'mean_homepage', 'mean_oucontent', 'mean_resource',
       'mean_subpage', 'mean_url', 'sum_forumng', 'sum_homepage',
       'sum_oucontent', 'sum_resource', 'sum_subpage', 'sum_url']]    

    if check_course_filter:
        if check_student_filter:
            chart_data = df_latest_semester1[(df_latest_semester1['code_module'] == filter_course) & (df_latest_semester1['id_student'] == filter_student)]
            chart_data = chart_data.append(df_latest_semester1.agg(['mean'])) 
            chart_data.rename(index={'mean':0},inplace=True)
            chart_data.iloc[[1],[0]] = 0
            chart_data.iloc[[1],[1]] = 'Average'
            st.dataframe(chart_data)
            chart_data1 = chart_data.groupby('id_student').first()
            chart_data1.index.name = 'id_student'
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.bar_chart(chart_data1[['mean_forumng']])            
            with col2:
                st.bar_chart(chart_data1[['mean_homepage']]) 
            with col3:
                st.bar_chart(chart_data1[['mean_oucontent']])    
            with col4:
                st.bar_chart(chart_data1[['mean_resource' ]])
        else:
            chart_data = df_latest_semester1[df_latest_semester1['code_module'] == filter_course]
            st.dataframe(chart_data,height=400)
            chart_data1 = chart_data.groupby('id_student').first().head(50)
            chart_data1.index.name = 'id_student'
                
            st.bar_chart(chart_data1[['mean_forumng']])            
            st.bar_chart(chart_data1[['mean_homepage']]) 
            st.bar_chart(chart_data1[['mean_oucontent']])    
            st.bar_chart(chart_data1[['mean_resource' ]])


            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="mean_forumng")
                st.pyplot(fig)            
            
            with col2:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="mean_homepage")
                st.pyplot(fig) 


            with col1:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="mean_oucontent")
                st.pyplot(fig)            
            
            with col2:
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(chart_data.head(50), x="mean_resource")
                st.pyplot(fig) 


    elif check_student_filter:
        st.dataframe(df_latest_semester1[ df_latest_semester1['id_student'] == filter_student])
    else:
        st.dataframe(df_latest_semester1,height=400)

        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="mean_forumng", bins=10)
            st.pyplot(fig)            
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="sum_forumng", bins=10)
            st.pyplot(fig)              



        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="mean_homepage",bins=20)
            st.pyplot(fig)            
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="sum_homepage", bins=20)
            st.pyplot(fig)            
        

        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="mean_oucontent")
            st.pyplot(fig)           
        
        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="sum_oucontent")
            st.pyplot(fig) 


        with col1:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="mean_resource")
            st.pyplot(fig)             

        with col2:
            fig = plt.figure(figsize=(10, 4))
            sns.histplot(df_latest_semester1, x="sum_resource")
            st.pyplot(fig) 


# Predicted students that will fail or drop out
if (selected == "At-Risk Students"):
    
    # page title
    st.title("At-Risk Students Prediction using ML")
    df_latest_semester = df[df['code_presentation'] == '2014J']

        # getting the input data from the user
    col1, col2, col3, col4, col5= st.columns(5)

    with col1:
        check_course_filter = st.checkbox("Filter on Course",value = False,key=1)
    with col2:
        filter_course = st.selectbox('Select Course', df_latest_semester['code_module'].unique())    
    with col3:
        check_student_filter = st.checkbox("Filter on Student",value = False,key=1)
    with col4:
        if check_course_filter:
            df_latest_semester = df_latest_semester[df_latest_semester['code_module'] == filter_course]
        else:
            df_latest_semester = df[df['code_presentation'] == '2014J'] 


        filter_student = st.selectbox('Filter on Student', df_latest_semester['id_student'])
        
    with col5:
        only_at_risk_filter = st.checkbox("Only View At-Risk Students",value = False,key=1)

    
    label_encoder = preprocessing.LabelEncoder() 
    mapping_list = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'age_band', 'disability','final_result']

    prediction_df = df_latest_semester.copy()
    # map labels to numbers
    for f in mapping_list: 
        prediction_df[f]= label_encoder.fit_transform(prediction_df[f]) 
        mapping = dict(zip(label_encoder.classes_, range(0, len(label_encoder.classes_)+1)))


    feature_names1 = ['code_module',  'gender', 'region',
       'highest_education', 'age_band', 'num_of_prev_attempts',
       'studied_credits', 'disability',  'score_count',
       'score_mean', 'score_sum', 'submission_delay_mean',
       'submission_delay_sum', 'weighted_score_mean', 'weighted_score_sum',
       'mean_forumng', 'mean_homepage', 'mean_oucontent', 'mean_resource',
       'mean_subpage', 'mean_url', 'sum_forumng', 'sum_homepage',
       'sum_oucontent', 'sum_resource', 'sum_subpage', 'sum_url']
    
    X = prediction_df[feature_names1]
    y = prediction_df['final_result']
    
    predictions = prediction_model.predict(X)
    df_latest_semester['predicted_outcome'] = predictions
    # df_latest_semester['Suggested_Intervention'] = predictions

    df_latest_semester.loc[df_latest_semester.predicted_outcome == 0, 'Suggested_Intervention'] = "Conduct Face to Face Video Session"
    df_latest_semester.loc[df_latest_semester.predicted_outcome == 1, 'Suggested_Intervention'] = "No Intervention Needed"
    chart_data = df_latest_semester

    if check_course_filter:
        if check_student_filter:
            chart_data = df_latest_semester[(df_latest_semester['code_module'] == filter_course) & (df_latest_semester['id_student'] == filter_student)]
        else:
            chart_data = df_latest_semester[df_latest_semester['code_module'] == filter_course]
    elif check_student_filter:
        chart_data = df_latest_semester[ df_latest_semester['id_student'] == filter_student]
    else:
        chart_data = df_latest_semester
       
    if (only_at_risk_filter):
        chart_data = chart_data[ chart_data['predicted_outcome'] == 0]
    
    st.write(chart_data[['id_student',  'predicted_outcome', 'Suggested_Intervention']])


# At-Risk Students Manual Prediction Page
if (selected == "Manual Prediction"):
    
    # page title
    st.title("At-Risk Students Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        code_module = st.selectbox('Select Subject', df['code_module'].unique())

    with col2:
        gender = st.selectbox('Select Gender', df['gender'].unique())
        
    with col3:
        highest_education = st.selectbox('Select Highest Education', df['highest_education'].unique())
        
    with col4:
        age_band = st.selectbox('Select Age Band', df['age_band'].unique())
        
    with col5:
        region = st.selectbox('Select Region', df['region'].unique())

    with col1:
        studied_credits = st.text_input('studied_credits')
        
    with col2:
        disability = st.text_input('disability')
        
    with col3:
        score_count = st.text_input('score_count')
        
    with col4:
        score_mean = st.text_input('score_mean')
        
    with col5:
        score_sum = st.text_input('score_sum')
          
    with col1:
        submission_delay_mean = st.text_input('submission_delay_mean')
        
    with col2:
        submission_delay_sum = st.text_input('submission_delay_sum')
        
    with col3:
        weighted_score_mean = st.text_input('weighted_score_mean')
        
    with col4:
        weighted_score_sum = st.text_input('weighted_score_sum')
        
    with col5:
        mean_forumng = st.text_input('mean_forumng')
          
    with col1:
        mean_homepage = st.text_input('mean_homepage')
        
    with col2:
        mean_oucontent = st.text_input('mean_oucontent')
        
    with col3:
        mean_resource = st.text_input('mean_resource')
        
    with col4:
        mean_subpage = st.text_input('mean_subpage')
        
    with col5:
        mean_url = st.text_input('mean_url')
           
    with col1:
        sum_forumng = st.text_input('sum_forumng')
        
    with col2:
        sum_homepage = st.text_input('sum_homepage')
        
    with col3:
        sum_oucontent = st.text_input('sum_oucontent')
        
    with col4:
        sum_resource = st.text_input('sum_resource')
        
    with col5:
        sum_subpage = st.text_input('sum_subpage')
          
    with col1:
        sum_url = st.text_input('sum_url')
        
    with col2:
        num_of_prev_attempts = st.text_input('num_of_prev_attempts')
        

    course_codes = {'AAA': 0, 'BBB': 1, 'CCC': 2, 'DDD': 3, 'EEE': 4, 'FFF': 5, 'GGG': 6}
    gender_codes = {'F': 0, 'M': 1}
    region_codes = {'East Anglian Region': 0, 'East Midlands Region': 1, 'Ireland': 2, 'London Region': 3, 'North Region': 4, 'North Western Region': 5, 'Scotland': 6, 'South East Region': 7, 'South Region': 8, 'South West Region': 9, 'Wales': 10, 'West Midlands Region': 11, 'Yorkshire Region': 12}
    highest_education_codes = {'A Level or Equivalent': 0, 'HE Qualification': 1, 'Lower Than A Level': 2, 'No Formal quals': 3, 'Post Graduate Qualification': 4}
    age_codes = {'0-35': 0, '35-55': 1, '55<=': 2}


    std_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Get Student Result Prediction'):
        df = pd.DataFrame([[course_codes[code_module],  gender_codes[gender], region_codes[region],
        highest_education_codes[highest_education], age_codes[age_band], num_of_prev_attempts,
        studied_credits, disability,  score_count,
        score_mean, score_sum, submission_delay_mean,
        submission_delay_sum, weighted_score_mean, weighted_score_sum,
        mean_forumng, mean_homepage, mean_oucontent, mean_resource,
        mean_subpage, mean_url, sum_forumng, sum_homepage,
        sum_oucontent, sum_resource, sum_subpage, sum_url]], 
        columns=('code_module',  'gender', 'region',
       'highest_education', 'age_band', 'num_of_prev_attempts',
       'studied_credits', 'disability',  'score_count',
       'score_mean', 'score_sum', 'submission_delay_mean',
       'submission_delay_sum', 'weighted_score_mean', 'weighted_score_sum',
       'mean_forumng', 'mean_homepage', 'mean_oucontent', 'mean_resource',
       'mean_subpage', 'mean_url', 'sum_forumng', 'sum_homepage',
       'sum_oucontent', 'sum_resource', 'sum_subpage', 'sum_url'))

        st.dataframe(df)
        st.write(df.info())
        std_prediction = prediction_model.predict(df)

        if (std_prediction[0] == 1):
            std_diagnosis = 'The Student Will Pass'
        else:
            std_diagnosis = 'The Student Will Fail'
        
    st.success(std_diagnosis)



# Course Recommendations Page
if (selected == "Course Recommendations"):
    
    # page title
    st.title("Course Recommendations using ML Tech")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
