import pandas as pd
import spacy
import re
from collections import Counter, defaultdict
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS

# Load SpaCy
nlp = spacy.load('de_core_news_sm', disable=["parser", "ner"])

# Combine stopwords
CUSTOM_STOPWORDS = {'\n', '\n\n', '&amp;', 'fridays4future', 'fridaysforfuture',
                    '#FridaysForFuture', '#fridays4future', '#fridaysforfuture'}
ALL_STOPWORDS = nlp.Defaults.stop_words.union(GENSIM_STOPWORDS).union(CUSTOM_STOPWORDS)

# ---------- TEXT CLEANING ----------

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"RT\s+", "", text)
    return text

def tokenize_and_lemmatize(texts):
    lemma_tokens = []
    for doc in nlp.pipe(texts, batch_size=1000):
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and not token.is_punct and token.pos_ != "PRON"]
        tokens = [re.sub(r'\W+', '', tok) for tok in tokens if tok not in ALL_STOPWORDS and len(tok) > 1]
        lemma_tokens.append(tokens)
    return lemma_tokens

# ---------- TOPIC MODELING ----------

def run_lda_model(df_tokens, min_topics=2, max_topics=15):
    id2word = Dictionary(df_tokens)
    id2word.filter_extremes(no_below=2, no_above=0.99)
    corpus = [id2word.doc2bow(text) for text in df_tokens]

    max_topics = min(max_topics, len(df_tokens))  # Don't exceed data size

    best_model = None
    best_score = 0
    best_topics = []
    for num_topics in range(min_topics, max_topics + 1):
        model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word, random_state=42, iterations=70)
        coherence = CoherenceModel(model=model, texts=df_tokens, dictionary=id2word, coherence='c_v').get_coherence()
        if coherence > best_score:
            best_score = coherence
            best_model = model
            best_topics = model.print_topics()

    topics_clean = [' '.join(re.findall(r'"([^"]*)"', t[1])) for t in best_model.print_topics()]
    return {
        'topics': topics_clean,
        'num_topics': len(topics_clean),
        'coherence': best_score
    }

# ---------- GROUPED ANALYSIS ----------

def compute_group_topics(df, group_col, out_path):
    results = []
    groups = df[group_col].unique()
    for group in tqdm(groups, desc=f"Modeling by {group_col}"):
        subset = df[df[group_col] == group]
        if len(subset) < 3:
            continue
        result = run_lda_model(subset['lemma_tokens'].tolist())
        results.append({
            'group': group,
            'n_docs': len(subset),
            'num_topics': result['num_topics'],
            'coherence': result['coherence'],
            'topics': result['topics']
        })
    topic_df = pd.DataFrame(results)
    topic_df['topics_readable'] = topic_df['topics'].apply(lambda t: '\n'.join([f"Topic {i+1}: {w}" for i, w in enumerate(t)]))
    topic_df.to_csv(out_path, index=False)
    return topic_df

# ---------- MAIN ----------

def main():
    df = pd.read_csv('/Users/xixuan/Desktop/twitter_test/fff_api_alltweets/tweet_df_classified14.csv')
    df_meta = pd.read_csv('/Users/xixuan/Desktop/twitter_test/fff_api_alltweets/diff_gephi_sto_cross.csv')
    df['community'] = df_meta['community']
    df['label'] = df['label_manual']
    df['subcomm'] = df['sub'] + df['community']

    # Group by subcomm
    sub_groups = df.groupby('subcomm')['content'].apply(lambda x: ' '.join(x.dropna().unique()))
    df_clean = pd.DataFrame({
        'group': sub_groups.index,
        'content': sub_groups.values
    })
    df_clean['label'] = df_clean['group'].map(df.drop_duplicates('subcomm').set_index('subcomm')['label'])
    df_clean['comm'] = df_clean['group'].map(df.drop_duplicates('subcomm').set_index('subcomm')['community'])
    df_clean['commlabel'] = df_clean['comm'] + df_clean['label']

    # Clean + tokenize
    df_clean['content'] = df_clean['content'].apply(clean_text)
    df_clean['lemma_tokens'] = tokenize_and_lemmatize(df_clean['content'].tolist())

    # Save intermediate
    df_clean.to_csv('/Users/xixuan/Desktop/twitter_test/fff_api_alltweets/lda_cleaned.csv', index=False)

    # Topic modeling by subcomm
    compute_group_topics(df_clean, 'group', '/Users/xixuan/Desktop/twitter_test/fff_api_alltweets/topicmodels_subcomm.csv')

    # By label
    compute_group_topics(df_clean, 'label', '/Users/xixuan/Desktop/twitter_test/fff_api_alltweets/topicmodels_label.csv')

    # By community
    compute_group_topics(df_clean, 'comm', '/Users/xixuan/Desktop/twitter_test/fff_api_alltweets/topicmodels_comm.csv')


if __name__ == "__main__":
    main()
