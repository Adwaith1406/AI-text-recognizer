import os
import re
import string
import numpy as np
import pandas as pd
import spacy
import textstat
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "train_v2_drcat_02.csv"
CACHE_FILE = "extracted_features_cache.parquet"
SAMPLE_SIZE = 1200  # 600 Human vs 600 AI for fast, balanced extraction on Mac

print("[*] Loading spaCy and VADER models...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])
vader = SentimentIntensityAnalyzer()
contraction_re = re.compile(r"\b([a-zA-Z]+)('nt|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)

# ==========================================
# FEATURE EXTRACTION (PAPER TABLE 2)
# ==========================================
def extract_features(doc, text: str) -> dict:
    words = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
    word_count = max(len(words), 1)
    char_count = len(text)
    
    # 1. Lexical Retrieval Features
    unique_words = set(words)
    ttr = len(unique_words) / word_count
    
    freqs = {}
    for w in words:
        freqs[w] = freqs.get(w, 0) + 1
    hapax_rate = sum(1 for w, c in freqs.items() if c == 1) / word_count
    avg_word_len = char_count / word_count
    
    # 2. Syntactic & Cognitive Load Features
    sentences = list(doc.sents)
    sent_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sent_count
    
    punct_rate = sum(1 for c in text if c in string.punctuation) / word_count
    contraction_rate = len(contraction_re.findall(text)) / word_count
    
    complex_sentences = 0
    for sent in sentences:
        clauses = sum(1 for tok in sent if tok.dep_ in ('advcl', 'ccomp', 'xcomp', 'relcl'))
        if clauses >= 1:
            complex_sentences += 1
    complex_ratio = complex_sentences / sent_count
    
    # 3. Sentiment & Subjectivity
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    vader_compound = vader.polarity_scores(text)['compound']
    
    # 4. Readability Indices
    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        gunning_fog = textstat.gunning_fog(text)
    except:
        flesch_reading_ease = 50.0
        gunning_fog = 10.0
        
    first_person_pronouns = {'i', 'me', 'my', 'myself', 'we', 'our', 'us', 'ours'}
    first_person_rate = sum(1 for w in words if w in first_person_pronouns) / word_count
    
    return {
        "word_count": word_count,
        "ttr": ttr,
        "hapax_rate": hapax_rate,
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "punct_rate": punct_rate,
        "contraction_rate": contraction_rate,
        "complex_ratio": complex_ratio,
        "subjectivity": subjectivity,
        "vader_compound": vader_compound,
        "flesch_reading_ease": flesch_reading_ease,
        "gunning_fog": gunning_fog,
        "first_person_rate": first_person_rate
    }

# ==========================================
# LOCAL DATA PREPARATION
# ==========================================
def load_data():
    if os.path.exists(CACHE_FILE):
        print(f"[*] Loading processed feature cache from '{CACHE_FILE}'...")
        df = pd.read_parquet(CACHE_FILE)
        return df.drop(columns=['label']), df['label'].values

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Could not find '{CSV_FILE}' in the project directory!\n"
            f"Please download the dataset from Kaggle and move '{CSV_FILE}' into this folder."
        )

    print(f"[*] Reading '{CSV_FILE}'...")
    df = pd.read_csv(CSV_FILE)
    
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'label' if 'label' in df.columns else 'generated'
    
    df = df.dropna(subset=[text_col, label_col])
    df['word_len'] = df[text_col].apply(lambda x: len(str(x).split()))
    df = df[df['word_len'] >= 35]
    
    half = SAMPLE_SIZE // 2
    df_human = df[df[label_col] == 0].sample(min(half, (df[label_col] == 0).sum()), random_state=42)
    df_ai = df[df[label_col] == 1].sample(min(half, (df[label_col] == 1).sum()), random_state=42)
    sample_df = pd.concat([df_human, df_ai]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"[*] Extracting psycholinguistic features for {len(sample_df)} essays via spaCy batching...")
    raw_texts = sample_df[text_col].tolist()
    labels = sample_df[label_col].astype(int).values
    
    features = []
    for doc, text in zip(nlp.pipe(raw_texts, batch_size=200), raw_texts):
        features.append(extract_features(doc, text))
        
    feat_df = pd.DataFrame(features)
    feat_df['label'] = labels
    
    # Save cache to avoid re-extracting on future runs
    feat_df.to_parquet(CACHE_FILE)
    print(f"[+] Feature matrix saved to '{CACHE_FILE}'.")
    
    return feat_df.drop(columns=['label']), labels

# ==========================================
# TRAINING & TERMINAL TESTER
# ==========================================
def main():
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[*] Training Gaussian Naive Bayes (with Yeo-Johnson transform)...")
    model = Pipeline([
        ('transformer', PowerTransformer(method='yeo-johnson')),
        ('classifier', GaussianNB(var_smoothing=1e-8))
    ])
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[+] Training complete. Test Accuracy: {acc * 100:.2f}%\n")
    
    print("=" * 60)
    print(" PSYCHOLINGUISTIC AI-TEXT DETECTOR (Live Terminal)")
    print(" Enter or paste an essay/paragraph (30+ words).")
    print(" Type 'exit' to quit.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("Enter text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit'):
                break
                
            wc = len(user_input.split())
            if wc < 25:
                print(f"\n[!] Notice: Input has only {wc} words.")
                print("    Stylometric metrics (TTR, clause depth, readability) require")
                print("    extended text to gauge natural cognitive variance.\n")
                
            doc = nlp(user_input)
            feats = pd.DataFrame([extract_features(doc, user_input)])
            
            pred = model.predict(feats)[0]
            probs = model.predict_proba(feats)[0]
            
            label_name = "AI-GENERATED" if pred == 1 else "HUMAN-WRITTEN"
            print("-" * 40)
            print(f" PREDICTION : {label_name}")
            print(f" CONFIDENCE : {probs[pred] * 100:.2f}%")
            print(f" Likelihood : Human: {probs[0]*100:.1f}% | AI: {probs[1]*100:.1f}%")
            #print(f" TTR (Lexical Retrieval)  : {feats['ttr'].values[0]:.3f}")
            #print(f" Gunning Fog (Complexity) : {feats['gunning_fog'].values[0]:.2f}")
            #print(f" Subjectivity (Stance)    : {feats['subjectivity'].values[0]:.2f}")
            print("-" * 40 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()