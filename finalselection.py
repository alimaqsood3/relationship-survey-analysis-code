#!/usr/bin/env python
# coding: utf-8

# In[85]:


#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    print("Ensure you have installed the model with 'python -m spacy download en_core_web_sm'")

# 1. Data Loading
try:
    df = pd.read_excel('survey-relationship copy.xlsx')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File 'survey-relationship copy.xlsx' not found. Please check the file path.")
except Exception as e:
    print(f"Error loading dataset: {e}")

print(df.head())
print(df.info())

# 2. Data Cleaning and Preprocessing
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)

text_column = 'What happened or what did not happen in the past few weeks, that might have affected your satisfaction positively or negatively?'
df['cleaned_text'] = df[text_column].apply(clean_text)
df['preprocessed_text'] = df['cleaned_text'].apply(preprocess_text)

# 3. Feature Engineering
# 3.1 Sentiment Analysis
sid = SentimentIntensityAnalyzer()
df['vader_sentiment'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['textblob_sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 3.2 Named Entity Recognition
def extract_entities(text):
    doc = nlp(text)
    return [ent.label_ for ent in doc.ents]

df['entities'] = df['cleaned_text'].apply(extract_entities)

# 3.3 Word Embeddings
sentences = df['preprocessed_text'].apply(lambda x: x.split())
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
df['word_embeddings'] = df['preprocessed_text'].apply(lambda x: np.mean([word2vec_model.wv[word] for word in x.split() if word in word2vec_model.wv.key_to_index], axis=0))

# 3.4 TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['preprocessed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# 3.5 Topic Modeling
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_output = lda_model.fit_transform(tfidf_matrix)
topic_names = [f'Topic_{i+1}' for i in range(lda_model.n_components)]
df[topic_names] = lda_output

# 4. Handling Categorical Variables
categorical_columns = ['OS', 'Country', 'Area', 'Gender', 'Age', 'Marital Status', 'Education', 'Employment Status']
df_encoded = pd.get_dummies(df[categorical_columns], columns=categorical_columns)

# 5. Handle Missing Values
numeric_columns = df.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 6. Feature Scaling
numerical_features = ['vader_sentiment', 'textblob_sentiment'] + topic_names
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 7. Combine Features
def ensure_iterable(x, expected_length=100):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return x
    else:
        return [0] * expected_length

df['word_embeddings'] = df['word_embeddings'].apply(ensure_iterable)
word_embeddings_df = pd.DataFrame(df['word_embeddings'].tolist())
word_embeddings_df = word_embeddings_df.fillna(0)

final_features = pd.concat([
    df[['ID'] + numerical_features],
    df_encoded.reset_index(drop=True),
    word_embeddings_df,
    tfidf_df
], axis=1)
final_features.columns = final_features.columns.astype(str)

# 8. Feature Selection
X = final_features.drop('ID', axis=1)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'])

selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X, y)
selected_feature_names = X.columns[selector.get_support()]

final_features = final_features[['ID'] + list(selected_feature_names)]

# 9. Analyze Target Variable Distribution
plt.figure(figsize=(10, 6))
pd.Series(y).value_counts().plot(kind='bar')
plt.title('Distribution of Relationship Satisfaction')
plt.xlabel('Satisfaction Level')
plt.ylabel('Count')
plt.show()

# 10. Correlation Analysis
correlation_matrix = final_features.drop('ID', axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# 11. Dimensionality Reduction (PCA)
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_selected)
print(f"Number of components after PCA: {X_pca.shape[1]}")

# 12. Handle Imbalanced Data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 13. Model Selection and Training
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)  # Added XGBoost
}

# Iterate through each model for evaluation
for name, model in models.items():
    print(f"\n{name}:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train and evaluate on test set
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        print(f"\nROC AUC Score: {roc_auc:.4f}")

# 14. Save Preprocessed Data
final_features.to_csv('preprocessed_survey_data_advanced.csv', index=False)
print("\nAdvanced data preprocessing completed. Preprocessed data saved to 'preprocessed_survey_data_advanced.csv'")
print(f"Final feature set shape: {final_features.shape}")
print(f"Selected features: {', '.join(selected_feature_names[:10])}...")  # Showing first 10 features


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer

# Prepare Label Binarizer for ROC Curve plotting
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# Iterate through each model for evaluation and visualization
for name, model in models.items():
    print(f"\n{name}:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train and evaluate on test set
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    # Ensure that y_test and y_pred are integers
    y_test_int = y_test.astype(int)
    y_pred_int = y_pred.astype(int)
    
    # Classification Report
    report = classification_report(y_test_int, y_pred_int, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Plot Classification Report
    plt.figure(figsize=(12, 6))
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']], annot=True, cmap='Blues', fmt='.2f')
    plt.title(f'Classification Report for {name}')
    plt.show()
    
    # Confusion Matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import LabelEncoder

    # Compute the confusion matrix
    cm = confusion_matrix(y_test_int, y_pred_int, labels=label_encoder.transform(label_encoder.classes_))

# Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Create a larger figure with higher DPI for better quality
    plt.figure(figsize=(14, 12), dpi=120)

# Plot the confusion matrix
    disp.plot(cmap='Blues', values_format='d')

# Increase font sizes for better readability
    plt.title(f'Confusion Matrix for {name}', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

# Adjust tick parameters
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

# Adjust layout to prevent overlap and ensure clarity
    plt.tight_layout()

# Show the plot
    plt.show()

    


    # ROC Curve
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        n_classes = y_pred_proba.shape[1]
        
        plt.figure(figsize=(12, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {name}')
        plt.legend(loc='lower right')
        plt.show()


# In[2]:


plt.figure(figsize=(10, 6))
plt.hist(df['vader_sentiment'], bins=20, alpha=0.5, label='VADER')
plt.hist(df['textblob_sentiment'], bins=20, alpha=0.5, label='TextBlob')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[4]:


from wordcloud import WordCloud

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights)
        ax.set_title(f'Topic {topic_idx+1}')
        ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

plot_top_words(lda_model, tfidf.get_feature_names_out(), 10, 'Topics in LDA model')


# In[5]:


from collections import Counter

all_entities = [entity for sublist in df['entities'] for entity in sublist]
entity_counts = Counter(all_entities)

plt.figure(figsize=(12, 6))
plt.bar(entity_counts.keys(), entity_counts.values())
plt.title('Most Common Entities in Survey Responses')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer

def extract_top_n_words_per_topic(model, feature_names, n_top_words):
    topic_keywords = []
    for topic_weights in model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_top_words]
        topic_keywords.append([feature_names[i] for i in top_keyword_locs])
    return topic_keywords

topic_keywords = extract_top_n_words_per_topic(lda_model, tfidf.get_feature_names_out(), 10)
for i, keywords in enumerate(topic_keywords):
    print(f"Topic {i+1}: {', '.join(keywords)}")


# In[7]:


topic_sentiments = []
for topic in range(lda_model.n_components):
    topic_docs = df[df[f'Topic_{topic+1}'] == df[[f'Topic_{i+1}' for i in range(lda_model.n_components)]].max(axis=1)]
    topic_sentiments.append(topic_docs['vader_sentiment'].mean())

plt.figure(figsize=(10, 6))
plt.bar(range(1, lda_model.n_components+1), topic_sentiments)
plt.title('Average Sentiment per Topic')
plt.xlabel('Topic')
plt.ylabel('Average Sentiment Score')
plt.show()


# In[8]:


from sklearn.manifold import TSNE

def plot_words(model, words):
    word_vectors = np.array([model.wv[w] for w in words if w in model.wv.key_to_index])
    tsne = TSNE(n_components=2, random_state=0)
    points = tsne.fit_transform(word_vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:, 0], points[:, 1], marker='o')
    for word, point in zip(words, points):
        plt.annotate(word, point)
    plt.show()

common_words = [word for word, count in word2vec_model.wv.key_to_index.items() if count > 10][:100]
plot_words(word2vec_model, common_words)


# In[9]:


from nrclex import NRCLex

def get_emotions(text):
    emotion_analyzer = NRCLex(text)
    emotions = emotion_analyzer.affect_frequencies
    return emotions

# Apply emotion detection to your preprocessed text
df['emotions'] = df['preprocessed_text'].apply(get_emotions)

# Extract individual emotion scores
emotion_categories = ['fear', 'anger', 'trust', 'surprise', 'sadness', 'disgust', 'joy', 'anticipation']
for emotion in emotion_categories:
    df[f'emotion_{emotion}'] = df['emotions'].apply(lambda x: x.get(emotion, 0))

# Visualize emotion distribution
plt.figure(figsize=(12, 6))
df[['emotion_' + e for e in emotion_categories]].mean().plot(kind='bar')
plt.title('Average Emotion Scores in Survey Responses')
plt.xlabel('Emotions')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze emotions by satisfaction level
satisfaction_levels = df['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'].unique()

plt.figure(figsize=(15, 8))
for emotion in emotion_categories:
    emotion_by_satisfaction = [df[df['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'] == level][f'emotion_{emotion}'].mean() for level in satisfaction_levels]
    plt.plot(satisfaction_levels, emotion_by_satisfaction, marker='o', label=emotion)

plt.title('Emotion Scores by Satisfaction Level')
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Emotion Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Correlation between emotions and sentiment
emotion_sentiment_corr = df[['vader_sentiment'] + ['emotion_' + e for e in emotion_categories]].corr()['vader_sentiment'].sort_values(ascending=False)
print("Correlation between emotions and VADER sentiment:")
print(emotion_sentiment_corr)

# Word cloud for highly emotional responses
from wordcloud import WordCloud

def generate_emotion_wordcloud(emotion):
    high_emotion_text = ' '.join(df[df[f'emotion_{emotion}'] > df[f'emotion_{emotion}'].quantile(0.75)]['preprocessed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(high_emotion_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for High {emotion.capitalize()} Responses')
    plt.show()

# Generate word clouds for each emotion
for emotion in emotion_categories:
    generate_emotion_wordcloud(emotion)


# In[12]:


import plotly.graph_objects as go

satisfaction_counts = df['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'].value_counts().sort_index()

fig = go.Figure(data=[go.Bar(
    x=satisfaction_counts.index,
    y=satisfaction_counts.values,
    marker_color='rgb(55, 83, 109)',
    text=satisfaction_counts.values,
    textposition='auto',
)])

fig.update_layout(
    title='Figure 1: Distribution of Relationship Satisfaction Levels',
    xaxis_title='Satisfaction Level',
    yaxis_title='Count',
    bargap=0.2,
    bargroupgap=0.1
)

fig.show()


# In[21]:


import plotly.express as px

age_gender_satisfaction = df.groupby(['Age', 'Gender'])['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'].value_counts(normalize=True).reset_index(name='Percentage')
age_gender_satisfaction['Percentage'] *= 100

fig = px.sunburst(age_gender_satisfaction, 
                  path=['Gender', 'Age', 'Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'], 
                  values='Percentage',
                  color='Percentage',
                  color_continuous_scale='RdBu',
                  title='Figure 2: Satisfaction Levels by Age and Gender')

fig.update_layout(width=800, height=800)
fig.show()
plt.figure(figsize=(12, 8))
age_gender_satisfaction = df.groupby(['Age', 'Gender'])['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?'].value_counts(normalize=True).unstack()

sns.heatmap(age_gender_satisfaction['Very satisfied'].unstack(), annot=True, fmt='.2%', cmap='YlGnBu')
plt.title('Figure 2: Satisfaction Levels (Very Satisfied) by Age and Gender')
plt.xlabel('Gender')
plt.ylabel('Age Group')
plt.tight_layout()
plt.show()



# In[30]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=('VADER Sentiment Scores', 'TextBlob Sentiment Scores'))

fig.add_trace(go.Histogram(x=df['vader_sentiment'], name='VADER', marker_color='#636EFA'), row=1, col=1)
fig.add_trace(go.Histogram(x=df['textblob_sentiment'], name='TextBlob', marker_color='#EF553B'), row=1, col=2)

fig.update_layout(title_text="Figure 3: Distribution of Sentiment Scores")
fig.update_xaxes(title_text="Sentiment Score", row=1, col=1)
fig.update_xaxes(title_text="Sentiment Score", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)

fig.show()


# In[23]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_top_words_plotly(model, feature_names, n_top_words, title):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Topic {i+1}' for i in range(5)])
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        row = topic_idx // 3 + 1
        col = topic_idx % 3 + 1
        
        fig.add_trace(
            go.Bar(y=top_features, x=weights, orientation='h', name=f'Topic {topic_idx+1}'),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Weight", row=row, col=col)
        fig.update_yaxes(title_text="Word", row=row, col=col)

    fig.update_layout(height=1000, width=1200, title_text=title, showlegend=False)
    fig.show()

plot_top_words_plotly(lda_model, tfidf.get_feature_names_out(), 10, 'Figure 4: Top Words for LDA Topics')


# In[24]:


emotion_cols = ['emotion_' + e for e in emotion_categories]
avg_emotions = df[emotion_cols].mean().sort_values(ascending=False)

fig = go.Figure(data=[go.Bar(
    x=avg_emotions.index,
    y=avg_emotions.values,
    marker_color=px.colors.sequential.Viridis[:len(avg_emotions)],
    text=[f'{val:.3f}' for val in avg_emotions.values],
    textposition='auto',
)])

fig.update_layout(
    title='Figure 5: Average Emotions in Survey Responses',
    xaxis_title='Emotion',
    yaxis_title='Average Score',
    xaxis_tickangle=-45
)

fig.show()


# In[25]:


emotion_by_satisfaction = df.groupby('Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?')[emotion_cols].mean()

fig = go.Figure()

for emotion in emotion_cols:
    fig.add_trace(go.Scatter(
        x=emotion_by_satisfaction.index,
        y=emotion_by_satisfaction[emotion],
        mode='lines+markers',
        name=emotion.split('_')[1]
    ))

fig.update_layout(
    title='Figure 6: Emotion Scores by Satisfaction Level',
    xaxis_title='Satisfaction Level',
    yaxis_title='Average Emotion Score',
    legend_title='Emotion'
)

fig.show()


# In[26]:


corr_features = ['vader_sentiment', 'textblob_sentiment'] + emotion_cols
corr_matrix = df[corr_features].corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmin=-1, zmax=1,
    text=corr_matrix.values,
    texttemplate='%{text:.2f}',
    textfont={"size": 10},
))

fig.update_layout(
    title='Figure 7: Correlation Heatmap of Key Features',
    width=800,
    height=800,
)

fig.show()


# In[28]:


import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np

# Assuming word2vec_model is your trained Word2Vec model
# Example: word2vec_model = Word2Vec.load("your_model_path")

# Select top N most common words
N = 100

# Get the top N words from the model's vocabulary
common_words = word2vec_model.wv.index_to_key[:N]
word_vectors = np.array([word2vec_model.wv[word] for word in common_words])

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(word_vectors)

# Plot with Plotly
fig = go.Figure(data=go.Scatter(
    x=tsne_results[:, 0],
    y=tsne_results[:, 1],
    mode='markers+text',
    text=common_words,
    textposition='top center',
    hoverinfo='text',
    marker=dict(size=10, color=tsne_results[:, 1], colorscale='Viridis', showscale=True)
))

fig.update_layout(
    title='Figure 8: t-SNE Visualization of Word Embeddings',
    xaxis_title='t-SNE feature 0',
    yaxis_title='t-SNE feature 1',
    width=1000,
    height=800
)

fig.show()


# In[29]:


from sklearn.ensemble import RandomForestClassifier

X = final_features.drop('ID', axis=1)
y = df['Please think about your spouse or partner. How satisfied are you with your relationship with him/her ?']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

import pandas as pd
import plotly.graph_objects as go

# Create a DataFrame with all features and their importances
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})

# Sort the features by importance and select the top 10
top_features = feature_importance.sort_values('importance', ascending=False).head(10)

# Plot with Plotly
fig = go.Figure(data=[go.Bar(
    x=top_features['importance'],
    y=top_features['feature'],
    orientation='h',
    marker_color='rgba(50, 171, 96, 0.6)',
    text=[f'{val:.4f}' for val in top_features['importance']],
    textposition='outside'
)])

fig.update_layout(
    title='Top 10 Feature Importances',
    xaxis_title='Importance',
    yaxis_title='Feature',
    width=1000,
    height=800
)

fig.show()



# In[32]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate
svm_pred = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[11]:


print(feature_importance.head())


# In[ ]:





# In[23]:


print(feature_importance)


# In[34]:


import pandas as pd

# Create DataFrame with correct feature names
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

# Print to check
print(feature_importance.head(11))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})
top_features = feature_importance.sort_values('importance').head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


import plotly.graph_objects as go

# Plot with Plotly
fig = go.Figure(data=[go.Bar(
    x=feature_importance['importance'],
    y=feature_importance['feature'],
    orientation='h',
    marker_color='rgba(50, 171, 96, 0.6)',
    text=[f'{val:.4f}' for val in top_10_features['importance']],
    textposition='outside'
)])

fig.update_layout(
    title='Top 10 Feature Importances',
    xaxis_title='Importance',
    yaxis_title='Feature',
    width=1000,
    height=800
)

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




