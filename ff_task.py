import json
import numpy as np
import pandas as pd
import random
import os.path
from nltk import sent_tokenize
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_extraction.text as sktext
from sklearn import metrics
from sklearn import svm
import requests
import urllib.request as urls
from urllib.error import HTTPError
try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def load_data():
    """Parse JSON file listing claims and their corresponding topics."""
    url = "https://fullfact.org/media/claim_conclusion.json"
    filename = CACHE_DIR + "/claim_conclusion.json"
    if not os.path.isfile(filename):
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)
    with open(filename) as file_in:
        data = file_in.readlines()
        checks = json.loads("".join(data[1:]))  # skip first line and parse JSON

    def get_topic(row):
        return(row['url'].split("/")[0])

    df = pd.DataFrame(checks)
    df['topic'] = df.apply(lambda row: get_topic(row), axis=1)
    return df


def get_web_text(df_url):
    """Get full text of claim from Full Fact website and store in local cache."""
    # e.g. get_web_text(df.iloc[2]['url'])
    url = "https://fullfact.org/" + df_url
    filename = CACHE_DIR + "/" + df_url.replace("/", "_")
    if os.path.isfile(filename):
        with open(filename) as file_in:
            text = file_in.read()
        return text
    try:
        html = urls.urlopen(url).read(50000)
        parsed_html = BeautifulSoup(html, 'lxml')
        lines = [p.text for p in parsed_html.find_all('p')]
        text = " ".join(lines)
        with open(filename, 'w') as outfile:
            outfile.write(text)
        return text
    except HTTPError as e:
        print("Web page error for url {}\n {}".format(url, e))
        with open(filename, 'w') as outfile:
            outfile.write("")
        return ""


def init_filter(df, report=False):
    """Keep relevant subset of topics"""
    keep_topics = ['economy', 'europe', 'health', 'crime', 'education', 'immigration', 'law']
    groups = df.groupby('topic')
    sub_df = df[df['topic'].isin(keep_topics)]
    sub_groups = sub_df.groupby('topic')
    if report:
        print("Found {} topics in data:".format(len(groups)))
        for g in groups:
            print("  {:20s} {:3d}".format(g[0], len(g[1])))
        print("Keeping {} topics".format(len(sub_groups)))
    return sub_df


def priors(df):
    """Number of documents in each topic"""
    groups = df.groupby('topic')
    priors = {g[0]: len(g[1]) for g in groups}
    return priors


def subsample(df, tightness=1, report=False):
    """Downsample larger topics to (roughly) size of smallest topic.
       Specify tightness < 1 to cap ratio of largest to smallest group produced."""
    groups = df.groupby('topic')
    smallest = min([len(g[1]) for g in groups])
    trim_to = int(smallest / tightness)

    df = df.groupby('topic').head(trim_to).reset_index(drop=True)
    groups = df.groupby('topic')
    print("Smallest group: {}; trimming others to {} items or fewer".format(smallest, trim_to))
    if report:
        print("New sizes of {} topics:".format(len(groups)))
        for g in groups:
            print("  {:20s} {:3d}".format(g[0], len(g[1])))
    return df


def train_test_split(df, ratio=0.8):
    df = df.sample(frac=1)  # shuffle collection before splitting
    split_point = int(len(df) * ratio)
    X_train = df['claim'].iloc[0:split_point]
    X_test = df['claim'].iloc[split_point:]
    y_train = df['topic'].iloc[0:split_point]
    y_test = df['topic'].iloc[split_point:]
    return ([X_train, X_test, y_train, y_test])


def enhance_data(df, X_train, y_train):
    """Get sentences from web pages to extend training set."""
    num_sentences = 20
    for i in range(0, len(df)):
        text = get_web_text(df.iloc[i]['url'])
        if text:
            ss = sent_tokenize(text)[0:num_sentences]
            topic = df.iloc[i]['topic']
            X_extra = pd.Series(ss)
            X_train = X_train.append(X_extra)
            y_extra = pd.Series([topic] * len(ss))
            y_train = y_train.append(y_extra)
    return X_train, y_train


def eval_model(X_train, X_test, y_train, y_test):
    """Build and evaluate a model"""
    count_vect = sktext.CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = sktext.TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    model = svm.LinearSVC()
    model.fit(X_train_tfidf, y_train)
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = model.predict(X_new_tfidf)
    df_out = pd.DataFrame(y_test)
    df_out['pred'] = predicted
    df_out['claim'] = X_test
    df_out['match'] = df_out['pred'] == df_out['topic']
    return (model, df_out)


def enhance_expt(df):
    """Compare model built with initial claim-only data to model built
       with extra data from FullFact.org"""
    X_train, X_test, y_train, y_test = train_test_split(df)
    model, df_out = eval_model(X_train, X_test, y_train, y_test)

    raw_f1 = metrics.f1_score(y_test, df_out['pred'], average=None)
    class_sizes = priors(df)
    weighted_f1 = np.dot(raw_f1, list(class_sizes.values())) / sum(class_sizes.values())
    results = {"init_size": len(X_train), "init_raw_f1": raw_f1, "init_weighted_f1": weighted_f1}

    X_train_en, y_train_en = enhance_data(df, X_train, y_train)
    model, df_out = eval_model(X_train_en, X_test, y_train_en, y_test)
    raw_f1 = metrics.f1_score(y_test, df_out['pred'], average=None)
    class_sizes = priors(df)
    weighted_f1 = np.dot(raw_f1, list(class_sizes.values())) / sum(class_sizes.values())
    results.update({"enhanced_size": len(X_train_en), "enhanced_raw_f1": raw_f1, "enhanced_weighted_f1": weighted_f1})
    return (results, df_out)


random.seed(1)
df_raw = load_data()
df = init_filter(df_raw)
df = subsample(df, 0.25)
results, df_out = enhance_expt(df)
print("Basic training size    {}\tF1={:.3f}".format(results["init_size"], results["init_weighted_f1"]))
print("Enahnced training size {}\tF1={:.3f}".format(results["enhanced_size"], results["enhanced_weighted_f1"]))

# num_errors = len(df_out[df_out['match'] is False])
# print("Total errors: {}".format(num_errors))
# if num_errors > 0:
#     print(df_out[df_out['match'] is False])
