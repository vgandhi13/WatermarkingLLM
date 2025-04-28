# watermarking
run this using a  venv so it doesn't mess with your alr installed packages
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch transformers nltk scikit-learn fasttext gensim joblib
```


to run glove:
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python convert_glove.py  ONLY RUN THIS ONCE BC IT WILL EAT UP ALL THE MEMORY
