from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import io
import nltk
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

app = Flask(__name__)

# folder graf
os.makedirs("static/graf", exist_ok=True)

# --- 1. LOAD DATA DARI PANDAS ---
def load_data():
    try:
        with open('beritaa.csv', 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')

        return pd.read_csv(
            io.StringIO(content),
            sep='\t',
            on_bad_lines='skip'
        ).fillna('')

    except:
        return pd.DataFrame()

df_berita = load_data()

@app.route('/')
def index():
    return render_template('index.html')

# --- 2. PENCARIAN BERITA FLEKSIBEL ---
@app.route('/search', methods=['POST'])
def cari():

    kw = request.json.get('keyword', '')
    pola = r'\b{}\b'.format(re.escape(kw))

    hasil = df_berita[
        df_berita['title'].str.contains(
            pola,
            case=False,
            na=False,
            regex=True
        )
    ].head(5)

    if hasil.empty:
        return jsonify({
            'error': 'Maaf, keyword ini tidak tersedia.'
        })

    return jsonify({
        'success': True,
        'data': hasil.to_dict('records')
    })

# --- 3. PROSES PERINGKASAN MULTI-DOKUMEN ---
@app.route('/summarize', methods=['POST'])
def ringkas():

    berita_list = request.json.get('berita', [])
    kw = request.json.get('keyword', '')

    # ================= PREPROCESSING =================
    teks = " ".join(berita_list)

    bersih = re.sub(
        r'SERAMBINEWS\.COM|TRIBUNNEWS\.COM|KOMPAS\.com|'
        r'SCROLL TO CONTINUE WITH CONTENT|'
        r'\[Gambas:.*?\]|'
        r'Baca juga:.*?(?=\.|$)|'
        r'\b[A-Z\s]{2,20}\s*[-,:]\s*',
        ' ',
        teks,
        flags=re.IGNORECASE
    )

    bersih = re.sub(r'\s+', ' ', bersih).strip()

    # pecah kalimat
    kalimat = [
        k.strip()
        for k in re.split(r'(?<=[.!?])\s+', bersih)
        if len(k.strip()) > 15
    ]

    if len(kalimat) == 0:
        return jsonify({
            'error': 'Maaf, keyword ini tidak tersedia.'
        })

    batas = 5 if len(kalimat) > 5 else len(kalimat)

    # ================= TEXTRANK =================

    # 1. TF-IDF
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(kalimat)

    # 2. Cosine Similarity
    sim_matrix = cosine_similarity(matrix)

    # 3. Pembentukan Graf
    nx_graph = nx.from_numpy_array(sim_matrix)

    # ================= VISUALISASI GRAF =================
    plt.figure(figsize=(10,7))

    pos = nx.spring_layout(nx_graph)

    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_color='skyblue',
        edge_color='gray',
        node_size=2000,
        font_size=8
    )

    plt.title(f"Graf Hubungan Antar Kalimat - {kw}")

    graf_path = f"static/graf/{kw}.png"

    plt.savefig(graf_path, bbox_inches='tight')
    plt.close()

    # 4. Perangkingan PageRank
    scores = nx.pagerank(nx_graph, alpha=0.85)

    # 5. Urutkan skor terbesar
    ranked_sentences_tr = sorted(
        ((scores[i], s) for i, s in enumerate(kalimat)),
        reverse=True
    )

    # 6. Ambil hasil ringkasan
    res_tr = " ".join([
        s for score, s in ranked_sentences_tr[:batas]
    ])

    # ================= LEXRANK =================

    teks_untuk_sumy = " ".join(kalimat)

    parser = PlaintextParser.from_string(
        teks_untuk_sumy,
        Tokenizer("english")
    )

    summarizer = LexRankSummarizer()

    summarizer.threshold = 0.2

    summary = summarizer(parser.document, batas)

    res_lr = " ".join([
        str(sentence)
        for sentence in summary
    ])

    # ================= HASIL =================
    return jsonify({
        'success': True,

        'textrank': {
            'ringkasan': res_tr
        },

        'lexrank': {
            'ringkasan': res_lr
        },

        'graf': f"/static/graf/{kw}.png"
    })

if __name__ == '__main__':
    app.run(debug=True)