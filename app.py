import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from collections import Counter
# from helper import perform_sentiment_analysis, count_word_occurrences

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
def categorize_sentiment(message):
    sentiment_score = analyzer.polarity_scores(message)['compound']
    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Function to read text file and perform sentiment analysis
def analyze_sentiment_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        messages = file.readlines()

    sentiment_counts = Counter()

    for message in messages:
        sentiment_category = categorize_sentiment(message)
        sentiment_counts[sentiment_category] += 1

    return sentiment_counts

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Read the text file
        file_path = 'message.csv'
        sentiment_counts = analyze_sentiment_from_file(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Tokenize the text into words
        words = word_tokenize(text)

        # Categorize words into positive, negative, or neutral
        positive_words = []
        negative_words = []
        neutral_words = []

        for word in words:
            sentiment_category = categorize_sentiment(word)
            if sentiment_category == 'Positive':
                positive_words.append(word)
            elif sentiment_category == 'Negative':
                negative_words.append(word)
            else:
                neutral_words.append(word)

        # Convert lists to strings for word cloud generation
        positive_text = ' '.join(positive_words)
        negative_text = ' '.join(negative_words)
        neutral_text = ' '.join(neutral_words)

        # Visualize the sentiment counts
        labels = sentiment_counts.keys()
        counts = sentiment_counts.values()

        st.subheader('Sentiment Analysis Results')
        # Define colors for bars
        colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

        # Visualize the sentiment counts with counts on bars
        fig, ax = plt.subplots()
        bars = ax.bar(sentiment_counts.keys(), sentiment_counts.values(),
                      color=[colors[sentiment] for sentiment in sentiment_counts.keys()])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')

        st.pyplot(fig)

        # Generate word clouds for each category with increased size
        wordcloud_positive = WordCloud(background_color='white', width=800, height=600).generate(positive_text)
        wordcloud_negative = WordCloud(background_color='white', width=800, height=600).generate(negative_text)
        wordcloud_neutral = WordCloud(background_color='white', width=800, height=600).generate(neutral_text)

        # Plot word clouds
        st.subheader('Sentiment Word Clouds')

        st.subheader('Positive Words')
        st.image(wordcloud_positive.to_array())
        st.subheader('Negative Words')
        st.image(wordcloud_negative.to_array())
        st.subheader('Neutral Words')
        st.image(wordcloud_neutral.to_array())


        # Function to check if a word contains only alphabetical characters
        def is_alpha(word):
            return word.isalpha()


        # Filter out non-alphabetical words
        positive_words_alpha = [word for word in positive_words if is_alpha(word)]
        negative_words_alpha = [word for word in negative_words if is_alpha(word)]
        neutral_words_alpha = [word for word in neutral_words if is_alpha(word)]

        # Count occurrences of each word in positive, negative, and neutral categories
        positive_word_counts = Counter(positive_words_alpha)
        negative_word_counts = Counter(negative_words_alpha)
        neutral_word_counts = Counter(neutral_words_alpha)

        # Display top 10 positive words
        st.subheader('Top 10 Positive Words')
        st.table(positive_word_counts.most_common(10))

        # Display top 10 negative words
        st.subheader('Top 10 Negative Words')
        st.table(negative_word_counts.most_common(10))

        # Display top 10 neutral words
        st.subheader('Top 10 Neutral Words')
        st.table(neutral_word_counts.most_common(10))


        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)
