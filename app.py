import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Emotion Analysis", layout="wide")
st.title("🎭 Emotion Analysis with ML - All Models")

# Load pre-trained models and vectorizers
@st.cache_resource
def load_models():
    with open('models.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    models = load_models()
except FileNotFoundError:
    st.error("❌ Models not found! Please run the notebook first to train and save the models.")
    st.stop()

# Sidebar configuration
st.sidebar.header("📊 Available Models")
all_models = ["Naive Bayes (BOW)", "Naive Bayes (TF-IDF)", "Logistic Regression (BOW)", "Logistic Regression (TF-IDF)"]
st.sidebar.info("Models available:\n" + "\n".join(f"• {m}" for m in all_models))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    new_text = ''
    for char in text:
        if char.isascii():
            new_text += char
    return new_text

# Function to predict with all models
def predict_all_models(text):
    processed_text = preprocess_text(text)
    
    results = {}
    
    # Naive Bayes BOW
    vec_bow = models['bow_vectorizer'].transform([processed_text])
    pred_nb_bow = models['model_nb_bow'].predict(vec_bow)[0]
    probs_nb_bow = models['model_nb_bow'].predict_proba(vec_bow)[0]
    results['Naive Bayes (BOW)'] = {
        'prediction': models['reverse_emotion'][pred_nb_bow],
        'probabilities': probs_nb_bow
    }
    
    # Naive Bayes TF-IDF
    vec_tfidf = models['tfidf_vectorizer'].transform([processed_text])
    pred_nb_tfidf = models['model_nb_tfidf'].predict(vec_tfidf)[0]
    probs_nb_tfidf = models['model_nb_tfidf'].predict_proba(vec_tfidf)[0]
    results['Naive Bayes (TF-IDF)'] = {
        'prediction': models['reverse_emotion'][pred_nb_tfidf],
        'probabilities': probs_nb_tfidf
    }
    
    # Logistic Regression BOW
    pred_lr_bow = models['model_lr_bow'].predict(vec_bow)[0]
    probs_lr_bow = models['model_lr_bow'].predict_proba(vec_bow)[0]
    results['Logistic Regression (BOW)'] = {
        'prediction': models['reverse_emotion'][pred_lr_bow],
        'probabilities': probs_lr_bow
    }
    
    # Logistic Regression TF-IDF
    pred_lr_tfidf = models['model_lr_tfidf'].predict(vec_tfidf)[0]
    probs_lr_tfidf = models['model_lr_tfidf'].predict_proba(vec_tfidf)[0]
    results['Logistic Regression (TF-IDF)'] = {
        'prediction': models['reverse_emotion'][pred_lr_tfidf],
        'probabilities': probs_lr_tfidf
    }
    
    return results

# Main tab interface
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Comparison", "Model Performance", "Batch Analysis"])

with tab1:
    st.subheader("🔍 Make a Prediction")
    user_input = st.text_area("Enter text to analyze emotion:", placeholder="Type something here...", height=120)
    
    if st.button("🚀 Predict with All Models", use_container_width=True):
        if user_input.strip():
            results = predict_all_models(user_input)
            
            # Show Logistic Regression results prominently at top
            st.subheader("⭐ Logistic Regression Results")
            lr_cols = st.columns(2)
            
            with lr_cols[0]:
                st.markdown("### Logistic Regression (BOW)")
                lr_bow_result = results['Logistic Regression (BOW)']
                st.metric("Emotion", lr_bow_result['prediction'], delta=f"{lr_bow_result['probabilities'].max()*100:.1f}% confidence")
                prob_df = pd.DataFrame({
                    'Emotion': [models['reverse_emotion'][i] for i in range(len(lr_bow_result['probabilities']))],
                    'Probability': [f"{p*100:.2f}%" for p in lr_bow_result['probabilities']]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            with lr_cols[1]:
                st.markdown("### Logistic Regression (TF-IDF)")
                lr_tfidf_result = results['Logistic Regression (TF-IDF)']
                st.metric("Emotion", lr_tfidf_result['prediction'], delta=f"{lr_tfidf_result['probabilities'].max()*100:.1f}% confidence")
                prob_df = pd.DataFrame({
                    'Emotion': [models['reverse_emotion'][i] for i in range(len(lr_tfidf_result['probabilities']))],
                    'Probability': [f"{p*100:.2f}%" for p in lr_tfidf_result['probabilities']]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Show all models comparison below
            st.divider()
            st.subheader("📊 All Models Comparison")
            
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Prediction': result['prediction'],
                    'Confidence': f"{result['probabilities'].max()*100:.2f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visualize predictions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Predictions Across Models")
                pred_counts = pd.Series([r['prediction'] for r in results.values()]).value_counts()
                st.bar_chart(pred_counts)
            
            with col2:
                st.markdown("### Confidence Scores")
                confidence_data = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Confidence': [r['probabilities'].max()*100 for r in results.values()]
                })
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(confidence_data['Model'], confidence_data['Confidence'], color='skyblue')
                ax.set_xlabel('Confidence (%)')
                ax.set_xlim(0, 100)
                for i, v in enumerate(confidence_data['Confidence']):
                    ax.text(v + 1, i, f'{v:.1f}%', va='center')
                st.pyplot(fig)
        else:
            st.warning("Please enter some text to analyze!")

with tab2:
    st.subheader("📊 Model Comparison Dashboard")
    
    if st.button("Generate Comparison Report", use_container_width=True):
        with st.spinner("Generating comparison report..."):
            comparison_metrics = []
            
            for model_name in all_models:
                if "BOW" in model_name:
                    if "Naive Bayes" in model_name:
                        model = models['model_nb_bow']
                        X_test_vec = models['X_test_bow']
                    else:
                        model = models['model_lr_bow']
                        X_test_vec = models['X_test_bow']
                else:
                    if "Naive Bayes" in model_name:
                        model = models['model_nb_tfidf']
                        X_test_vec = models['X_test_tfidf']
                    else:
                        model = models['model_lr_tfidf']
                        X_test_vec = models['X_test_tfidf']
                
                y_pred = model.predict(X_test_vec)
                accuracy = (y_pred == models['y_test']).mean()
                
                comparison_metrics.append({
                    'Model': model_name,
                    'Accuracy': f"{accuracy*100:.2f}%",
                    'Accuracy (decimal)': accuracy
                })
            
            comparison_df = pd.DataFrame(comparison_metrics)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(comparison_df[['Model', 'Accuracy']], use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### Accuracy Comparison")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(comparison_df['Model'], comparison_df['Accuracy (decimal)']*100, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
                ax.set_xlabel('Accuracy (%)')
                ax.set_xlim(0, 100)
                for i, v in enumerate(comparison_df['Accuracy (decimal)']*100):
                    ax.text(v + 1, i, f'{v:.2f}%', va='center')
                st.pyplot(fig)

with tab3:
    st.subheader("📈 Individual Model Performance")
    
    selected_model = st.selectbox("Select a model to view detailed performance:", all_models)
    
    if st.button("View Performance", use_container_width=True):
        with st.spinner("Computing performance metrics..."):
            # Map model name to model object and vectorizer
            if "BOW" in selected_model:
                if "Naive Bayes" in selected_model:
                    model = models['model_nb_bow']
                    X_test_vec = models['X_test_bow']
                else:
                    model = models['model_lr_bow']
                    X_test_vec = models['X_test_bow']
            else:
                if "Naive Bayes" in selected_model:
                    model = models['model_nb_tfidf']
                    X_test_vec = models['X_test_tfidf']
                else:
                    model = models['model_lr_tfidf']
                    X_test_vec = models['X_test_tfidf']
            
            y_pred = model.predict(X_test_vec)
            accuracy = (y_pred == models['y_test']).mean()
            
            # Display accuracy
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
            
            # Classification Report
            st.subheader("Classification Report")
            report = models['classification_reports'][selected_model]
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = models['confusion_matrices'][selected_model]
            cm_df = pd.DataFrame(cm, 
                                index=[models['reverse_emotion'][i] for i in range(len(cm))],
                                columns=[models['reverse_emotion'][i] for i in range(len(cm))])
            
            # Visualize confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
            ax.set_title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)

with tab4:
    st.subheader("📁 Batch Analysis")
    st.info("Upload a text file (one text per line) to analyze multiple texts at once")
    
    uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
    
    if uploaded_file is not None:
        lines = uploaded_file.read().decode('utf-8').split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if st.button("🚀 Analyze Batch with All Models", use_container_width=True):
            with st.spinner(f"Analyzing {len(lines)} texts..."):
                all_results = []
                
                for idx, line in enumerate(lines):
                    results = predict_all_models(line)
                    row_data = {
                        'Text': line[:50] + '...' if len(line) > 50 else line,
                    }
                    for model_name, result in results.items():
                        row_data[model_name] = result['prediction']
                    all_results.append(row_data)
                
                results_df = pd.DataFrame(all_results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                summary_cols = st.columns(len(all_models))
                
                for idx, model_name in enumerate(all_models):
                    with summary_cols[idx]:
                        emotion_counts = pd.Series(results_df[model_name]).value_counts()
                        st.markdown(f"### {model_name}")
                        for emotion, count in emotion_counts.items():
                            st.write(f"**{emotion}**: {count}")
                
                # Visualization
                st.subheader("📈 Predictions Distribution")
                model_cols = st.columns(2)
                for idx, model_name in enumerate(all_models):
                    with model_cols[idx % 2]:
                        emotion_counts = pd.Series(results_df[model_name]).value_counts()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f'{model_name}')
                        st.pyplot(fig)
