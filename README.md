# Hotel Customer Experience & Performance Analytics

Streamlit dashboard for:
- Gold-layer hotel performance and temporal analysis
- Aspect-Based Sentiment Analysis using saved scikit-learn models
- Hotel similarity network and Louvain community insights

## Expected artifact layout

```text
hotel-analytics-dashboard/
├── app.py
├── pages/
├── lib/
├── data/
│   ├── hotel_performance/       # Parquet part files OR a single .parquet file
│   ├── aspect_ratings/
│   └── temporal_trends/
├── models/
│   └── absa/
│       ├── tfidf_vectorizer.joblib
│       ├── logistic_regression.joblib
│       ├── linear_svm.joblib
│       └── labels.json
└── graph/
    ├── hotel_nodes.parquet
    ├── hotel_edges.parquet
    └── community_profiles.csv
```

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Push the application and the prepared artifacts to a GitHub repository, then
create a Streamlit Community Cloud app using `app.py` as the entry point.

For larger or private artifacts, use object storage or another accessible
artifact store rather than committing them to GitHub. Do not commit credentials
or API keys.

## Important notes

- Gold Parquet folders may contain Spark part files. The loader reads
  `part-*.parquet` and ignores hidden CRC files.
- The ABSA page uses the exact training input format:
  `aspect_<aspect> <review text>`.
- Linear SVM decision scores are not probabilities.
- The network page uses saved nodes and edges; it does not rerun Louvain.
- Confirm that the exported node Parquet contains `hotel_id`, `community`,
  and ideally `name`, `city`, `pct_positive`, and all six aspect columns.
