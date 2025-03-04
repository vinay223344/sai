import streamlit as st
from newspaper import Article
from transformers import pipeline

# Initialize the summarization model pipeline (BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def fetch_article(url):
    # Use newspaper3k library to fetch the article
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def clean_article_text(article_text):
    # Preprocess the article by removing unnecessary text (like boilerplate or ads).
    # This is a basic cleanup step; you can extend it further if needed.
    article_text = article_text.replace('\n', ' ').replace('  ', ' ').strip()
    return article_text

def summarize_article(article_text):
    # Clean the text before summarization
    article_text = clean_article_text(article_text)
    
    # Summarize the cleaned article text
    summary = summarizer(article_text, max_length=500, min_length=400, do_sample=False)
    return summary[0]['summary_text']

# Streamlit Interface
def main():
    st.title('News Article Summarizer')

    # URL Input by User
    url = st.text_input('Enter the URL of the news article:')

    if url:
        try:
            # Fetch the article text
            article_text = fetch_article(url)
            
            # Display the original article text (optional)
            # Display only first 1000 characters for brevity
            
            # Get the summary
            summary = summarize_article(article_text)
            
            # Display the summary
            st.subheader("Summary:")
            st.write(summary)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
