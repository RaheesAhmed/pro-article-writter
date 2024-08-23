import streamlit as st
import os
import asyncio
import requests
from openai import OpenAI
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
import textstat
import wordcloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import altair as alt
from streamlit_echarts import st_echarts
from streamlit_quill import st_quill
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import networkx as nx
from pyvis.network import Network
from st_aggrid import AgGrid
from gpt_researcher import GPTResearcher

# Streamlit UI setup
st.set_page_config(
    page_title="AI-Powered Research and Content Creation & Analysis Tool",
    page_icon="🔍",
    layout="wide",
)

# Load environment variables and initialize NLP tools
load_dotenv()
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")

# OpenAI API configuration
openai_client = OpenAI()
GPT_MODEL = "gpt-4-0125-preview"


# Function to load and inject custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load custom CSS
load_css("style.css")


# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Function to run asyncio functions in Streamlit
def run_async(func):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(func)


async def research_topic(query, report_type, sources, report_source):
    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        source_urls=sources if sources else None,
        report_source=report_source,
    )
    researcher.set_verbose(True)
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    return report, researcher


def generate_outline_and_statistics(research_data: str) -> str:
    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a highly experienced content creator specializing in technical and educational content. Your task is to generate a detailed outline, including relevant statistics and key points for an article",
            },
            {
                "role": "user",
                "content": f"Based on the following research data, carefully craft an outline that includes clear section headers, a logical flow of ideas, and any important statistics or facts that should be highlighted: {research_data}. Ensure the outline is structured to cover the topic comprehensively.",
            },
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def stream_article(outline_and_stats: str, words: int):
    stream = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a professional writer skilled in producing high-quality, long-form content. Your goal is to draft a detailed, engaging, and informative article based on the following outline",
            },
            {
                "role": "user",
                "content": f"Using the provided outline and statistics, please draft an in-depth article. The article should be structured logically, include well-supported arguments, and flow smoothly from one section to the next: {outline_and_stats}. Pay close attention to maintaining a professional tone and ensuring the content is informative and accessible. The article should be approximately {words} words long.",
            },
        ],
        temperature=0.7,
        max_tokens=4000,
        stream=True,
    )
    return stream


def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=5)
    return " ".join([str(sentence) for sentence in summary])


def analyze_text(text):
    # Word Frequency Analysis
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = Counter(words)

    # Sentence Length Distribution
    sentences = sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]

    # Named Entity Recognition
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Readability Scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)

    # Basic Sentiment Analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    # Topic Modeling
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])

    num_topics = 5
    n_samples = tfidf_matrix.shape[0]

    # Adjust the number of clusters if there are fewer samples
    if n_samples < num_topics:
        num_topics = n_samples

    if num_topics > 1:  # Ensure there's more than one cluster to form
        kmeans = KMeans(n_clusters=num_topics)
        kmeans.fit(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for i in range(num_topics):
            top_words = [feature_names[j] for j in kmeans.labels_.argsort()[-10:]]
            topics.append(", ".join(top_words))
    else:
        topics = ["Not enough data to perform topic modeling"]

    # Text Summarization using Sumy
    summary = summarize_text(text)

    # Advanced Sentiment Analysis using VADER
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)

    return {
        "word_freq": word_freq,
        "sentence_lengths": sentence_lengths,
        "entities": entities,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "sentiment": sentiment,
        "topics": topics,
        "summary": summary,
        "sentiment_scores": sentiment_scores,
    }


def create_visualizations(analysis_results):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="chart-header">Word Cloud</h3>', unsafe_allow_html=True)
        wc = wordcloud.WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(analysis_results["word_freq"])
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)

        st.markdown(
            '<h3 class="chart-header">Top 20 Words</h3>', unsafe_allow_html=True
        )
        word_freq_df = pd.DataFrame(
            analysis_results["word_freq"].most_common(20), columns=["word", "count"]
        )
        chart = (
            alt.Chart(word_freq_df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("word:N", sort="-x", title="Word"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["word", "count"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown(
            '<h3 class="chart-header">Sentence Length Distribution</h3>',
            unsafe_allow_html=True,
        )
        sentence_df = pd.DataFrame({"length": analysis_results["sentence_lengths"]})
        chart = (
            alt.Chart(sentence_df)
            .mark_area(
                line={"color": "darkblue"},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="white", offset=0),
                        alt.GradientStop(color="darkblue", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
            )
            .encode(
                alt.X(
                    "length:Q", bin=alt.Bin(maxbins=20), title="Sentence Length (words)"
                ),
                alt.Y("count()", title="Frequency"),
                tooltip=["length", "count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            '<h3 class="chart-header">Named Entity Types</h3>', unsafe_allow_html=True
        )
        entity_df = pd.DataFrame(
            analysis_results["entities"], columns=["Entity", "Type"]
        )
        entity_counts = entity_df["Type"].value_counts()
        pie_chart = {
            "tooltip": {"trigger": "item"},
            "legend": {"top": "5%", "left": "center"},
            "series": [
                {
                    "name": "Entity Types",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {"show": False, "position": "center"},
                    "emphasis": {
                        "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                    },
                    "labelLine": {"show": False},
                    "data": [{"value": v, "name": k} for k, v in entity_counts.items()],
                }
            ],
        }
        st_echarts(options=pie_chart, height="400px")

    # Topic Network Visualization
    G = nx.Graph()
    for i, topic in enumerate(analysis_results["topics"]):
        G.add_node(f"Topic {i+1}")
        for word in topic.split(", "):
            G.add_node(word)
            G.add_edge(f"Topic {i+1}", word)

    nt = Network(notebook=True, width="100%", height="500px")
    nt.from_nx(G)
    nt.show("topic_network.html")

    st.components.v1.html(open("topic_network.html", "r").read(), height=600)

    st.markdown(
        '<h3 class="chart-header">Readability and Sentiment Analysis</h3>',
        unsafe_allow_html=True,
    )
    col3, col4 = st.columns(2)

    with col3:
        readability_scores = {
            "Flesch Reading Ease": analysis_results["flesch_reading_ease"],
            "Flesch-Kincaid Grade": analysis_results["flesch_kincaid_grade"],
        }
        readability_df = pd.DataFrame(
            list(readability_scores.items()), columns=["Metric", "Score"]
        )
        chart = (
            alt.Chart(readability_df)
            .mark_bar()
            .encode(
                x=alt.X("Score:Q", title="Score"),
                y=alt.Y("Metric:N", title=""),
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="category10")),
                tooltip=["Metric", "Score"],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)

    with col4:
        sentiment = analysis_results["sentiment"]
        gauge_chart = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}"},
            "series": [
                {
                    "name": "Sentiment",
                    "type": "gauge",
                    "axisLine": {
                        "lineStyle": {
                            "width": 30,
                            "color": [
                                [-0.5, "#ff4500"],
                                [0.5, "#ffff00"],
                                [1, "#3fff00"],
                            ],
                        }
                    },
                    "pointer": {"itemStyle": {"color": "auto"}},
                    "axisTick": {
                        "distance": -30,
                        "length": 8,
                        "lineStyle": {"color": "#fff", "width": 2},
                    },
                    "splitLine": {
                        "distance": -30,
                        "length": 30,
                        "lineStyle": {"color": "#fff", "width": 4},
                    },
                    "axisLabel": {"color": "auto", "distance": 40, "fontSize": 15},
                    "detail": {
                        "valueAnimation": True,
                        "formatter": "{value}",
                        "color": "auto",
                    },
                    "data": [{"value": sentiment, "name": "Sentiment"}],
                    "min": -1,
                    "max": 1,
                }
            ],
        }
        st_echarts(options=gauge_chart, height="300px")


def display_entities(entities):
    df = pd.DataFrame(entities, columns=["Entity", "Type"])
    AgGrid(df)


def display_faq():
    st.markdown("## Frequently Asked Questions")

    faq_data = [
        {
            "question": "What is GPT Researcher & Analyzer?",
            "answer": "GPT Researcher & Analyzer is an AI-powered tool that conducts research on a given topic, generates an article, and provides in-depth analysis of the content including readability scores, sentiment analysis, and more.",
        },
        {
            "question": "How does the research process work?",
            "answer": "The tool uses advanced AI models to search for relevant information on the web or in specified sources, compiles a research report, generates an outline, and then writes a full article based on the gathered information.",
        },
        {
            "question": "Can I edit the generated article?",
            "answer": "Yes, the generated article is displayed in a rich text editor where you can make changes, add formatting, and even insert images. You can save your edits after making changes.",
        },
        {
            "question": "What kind of analysis is performed on the article?",
            "answer": "The tool performs various analyses including word frequency, sentence length distribution, named entity recognition, readability scoring, and sentiment analysis. These are presented in interactive visualizations.",
        },
        {
            "question": "How accurate is the generated content?",
            "answer": "While the AI strives for accuracy, it's always recommended to fact-check important information. The tool provides sources used in the research, which you can refer to for verification.",
        },
        {
            "question": "Can I use this for academic research?",
            "answer": "While this tool can be a great starting point for research, it's important to note that it should not be the sole source for academic work. Always verify information, cite original sources, and follow proper academic guidelines.",
        },
    ]

    for item in faq_data:
        with st.expander(item["question"]):
            st.write(item["answer"])


def main():
    # Lottie Animation
    lottie_research = load_lottieurl(
        "https://assets5.lottiefiles.com/packages/lf20_tno6cg2w.json"
    )
    st_lottie(lottie_research, speed=1, height=200)

    colored_header(
        label="AI-Powered Research and Content Creation",
        description="Generate high-quality articles and analyze content with advanced AI models",
        color_name="blue-70",
    )

    add_vertical_space(2)

    # Initialize session state variables
    if "full_article" not in st.session_state:
        st.session_state.full_article = ""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "show_editor" not in st.session_state:
        st.session_state.show_editor = False
    if "report" not in st.session_state:
        st.session_state.report = None
    if "researcher" not in st.session_state:
        st.session_state.researcher = None
    if "outline_and_stats" not in st.session_state:
        st.session_state.outline_and_stats = None

    # Input fields in a grid layout
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input(
            "Enter your research query:", "The impact of AI on modern education"
        )
    with col2:
        words = st.number_input(
            "Article Length (words)",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
        )
    with col3:
        report_type = st.selectbox(
            "Report Type",
            ["research_report", "resource_report", "outline_report", "custom_report"],
        )

    col4, col5 = st.columns([1, 3])
    with col4:
        report_source = st.selectbox("Source", ["web", "local", "hybrid"])
    with col5:
        sources = st.text_input(
            "Web sources (optional, comma-separated):",
            "https://en.wikipedia.org/wiki/Artificial_intelligence, https://www.edtechmagazine.com/",
        )

    # Button to start the research process
    if st.button("Start Research and Analysis"):
        with st.spinner("Researching and analyzing..."):
            # Step 1: Conduct research
            st.session_state.report, st.session_state.researcher = run_async(
                research_topic(
                    query=query,
                    report_type=report_type,
                    sources=sources.split(",") if sources else None,
                    report_source=report_source,
                )
            )

            # Step 2: Generate outline and statistics
            st.session_state.outline_and_stats = generate_outline_and_statistics(
                st.session_state.report
            )

            # Step 3: Generate article
            st.session_state.full_article = ""
            for chunk in stream_article(st.session_state.outline_and_stats, words):
                if chunk.choices[0].delta.content is not None:
                    st.session_state.full_article += chunk.choices[0].delta.content

            # Step 4: Analyze article
            st.session_state.analysis_results = analyze_text(
                st.session_state.full_article
            )

            # Set flag to show editor
            st.session_state.show_editor = True

        st.success("Research and analysis completed!")
        st.rerun()

    # Display results if available
    if st.session_state.show_editor:
        st.markdown(
            '<h2 class="section-header">Article Analysis</h2>', unsafe_allow_html=True
        )
        create_visualizations(st.session_state.analysis_results)

        st.markdown(
            '<h2 class="section-header">Generated Article</h2>', unsafe_allow_html=True
        )

        # Initialize the editor with the generated article
        content = st_quill(
            value=st.session_state.full_article,
            placeholder="Edit your article here...",
            html=True,
            key="quill",
            toolbar=[
                "bold",
                "italic",
                "underline",
                "strike",
                "header",
                "link",
                "image",
                "blockquote",
                "code-block",
                {"list": "ordered"},
                {"list": "bullet"},
                {"indent": "-1"},
                {"indent": "+1"},
            ],
        )

        # Add a button to save the edited content
        if st.button("Save Edited Article"):
            st.session_state.full_article = content
            st.success("Article saved successfully!")
            st.markdown("### Edited Article Preview")
            st.markdown(content, unsafe_allow_html=True)

        # Expandable sections
        if st.session_state.report is not None:
            with st.expander("Research Report"):
                st.write(st.session_state.report)
        if st.session_state.researcher is not None:
            with st.expander("Research Sources"):
                st.write(st.session_state.researcher.get_source_urls())
            with st.expander("Research Costs"):
                st.write(st.session_state.researcher.get_costs())
        if st.session_state.outline_and_stats is not None:
            with st.expander("Outline and Statistics"):
                st.write(st.session_state.outline_and_stats)

    # Display FAQ
    display_faq()

    # Footer
    st.markdown("---")
    st.markdown("Powered by GPT Researcher and OpenAI")


if __name__ == "__main__":
    main()
