import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

# Step 1: Load the data from the feedback survey
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Preprocess the text data
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]
    
    return ' '.join(words)


# Step 3: Visualize the data
def visualize_data(df):
    # Plot distribution of nine_box_category
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='nine_box_category')
    plt.title('Distribution of Nine Box Categories')
    plt.xlabel('Nine Box Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # Word cloud of processed feedback
    from wordcloud import WordCloud

    feedback_text = ' '.join(df['Processed Feedback'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Feedback')
    plt.show()


# Step 4: Identify common themes and areas for improvement
def identify_themes(df):
    from sklearn.feature_extraction.text import CountVectorizer

    # Vectorize processed feedback
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Processed Feedback'])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Sum up the counts of each word
    word_counts = X.sum(axis=0)

    # Convert to DataFrame
    word_freq = pd.DataFrame(word_counts, columns=feature_names).T
    word_freq.columns = ['Frequency']

    # Get top 10 most frequent words
    top_words = word_freq.sort_values(by='Frequency', ascending=False).head(10)

    print("Top 10 most frequent words:")
    print(top_words)

    return top_words.index.tolist()

# Step 5: Provide recommendations
def provide_recommendations(themes):
    print("Based on the identified themes, here are some recommendations:")
    for theme in themes:
        print(f"- Improve focus and time management to enhance productivity related to '{theme}'")
    

# Main function
def main():
    # Step 1: Load data
    file_path = 'employee_review.csv'
    feedback_data = load_data(file_path)
    
    # Step 2: Preprocess text data
    feedback_data['Processed Feedback'] = feedback_data['feedback'].apply(preprocess_text)
    
    # Step 3: Visualize data
    visualize_data(feedback_data)
    
    # Step 4: Identify common themes
    identified_themes = identify_themes(feedback_data)
    
    # Step 5: Provide recommendations
    provide_recommendations(identified_themes)

if __name__ == "__main__":
    main()
