import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as palm
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

load_dotenv()


API_KEY = 'AIzaSyD7eSg8_WIkx-URrLO0L9Fu6LYCL63Qh0Y'
palm.configure(api_key=API_KEY)

# Download the 'punkt' resource
nltk.download('punkt')

def generate_summary(text):
    if pd.notna(text) and isinstance(text, str):
        sentences = sent_tokenize(text)
        document = " ".join(sentences)
        parser = PlaintextParser.from_string(document, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count=5)
        return " ".join(str(sentence) for sentence in summary)
    else:
        return "Invalid input"

def generate_graphic_statistic(data):
    st.subheader("Graphic Analysis:")
    
    numeric_cols = data.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Add text labels to each bin
        for i, val in enumerate(data[col].value_counts().sort_index()):
            plt.text(i, val, str(val), ha='center', va='bottom')

        st.pyplot(plt)
        st.write(f"The above plot shows the distribution of {col}. Each bar represents a bin, and the number on top of each bar indicates the frequency of data points in that bin.")
        
        # Additional information about the analysis
        st.write(f"Analysis Summary for {col}:")
        st.write(f" - Mean: {data[col].mean()}")
        st.write(f" - Median: {data[col].median()}")
        st.write(f" - Standard Deviation: {data[col].std()}")

        # Save plot to BytesIO and provide a download button
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        st.download_button(label=f"Download {col} Distribution Plot", data=buffer, file_name=f"{col}_distribution_plot.png", key=f"{col}_distribution_plot")

def generate_descriptive_statistic(data):
    st.subheader("Descriptive Statistics:")
    
    stats = data.describe()
    
    # Additional statistics
    stats.loc['range'] = stats.loc['max'] - stats.loc['min']
    stats.loc['skew'] = data.skew()
    stats.loc['kurt'] = data.kurtosis()
    
    # Display the enhanced statistics
    st.dataframe(stats)
    
    # Additional statistics section
    st.subheader("Additional Statistics:")
    st.write(f" - Range: {stats.loc['range'].values}")
    st.write(f" - Skewness: {stats.loc['skew'].values}")
    st.write(f" - Kurtosis: {stats.loc['kurt'].values}")
    
    # Auto-generated conclusion
    conclusion = generate_conclusion(stats)
    st.subheader("Auto-Generated Conclusion:")
    st.write(conclusion)

    # Save statistics to CSV and provide a download button
    csv_data = stats.to_csv(index=True).encode()
    st.download_button(label="Download Descriptive Statistics", data=csv_data, file_name="descriptive_statistics.csv", key="descriptive_statistics")

def generate_conclusion(stats):
    conclusion = "Based on the descriptive statistics, we can make the following observations:\n"

    # Example conclusions (you can customize these based on your dataset and analysis)
    for col in stats.columns:
        if col != 'count':
            conclusion += f" - The {col} variable has a {get_tendency(stats.loc['mean'][col])} tendency.\n"

    return conclusion


def get_tendency(value):
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "neutral"


def main():

    st.title("Dataset Analysis Tool")

    # Form to upload the CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the dataset
        st.dataframe(df)

        # Select analysis type
        analysis_type = st.selectbox("Select Analysis Type", ['graphic_statistic', 'descriptive_statistic'])

        # Button to generate analysis
        if st.button("Generate Analysis"):
            # Perform the selected analysis
            if analysis_type == 'graphic_statistic':
                generate_graphic_statistic(df)
            elif analysis_type == 'descriptive_statistic':
                generate_descriptive_statistic(df)

            st.success(f"Analysis completed.")

# Run the app
if __name__ == "__main__":
    main()
