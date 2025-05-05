# watermarking
run this using a  venv so it doesn't mess with your alr installed packages
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch transformers nltk scikit-learn fasttext gensim joblib numpy
```


to run glove:
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python convert_glove.py  ONLY RUN THIS ONCE BC IT WILL EAT UP ALL THE MEMORY

Link to Presentation given for this project at UMass CS 692PA: https://docs.google.com/presentation/d/1iTQHRJhb3JFQLB4QdpOd7sjTvtm1eEE07NLCD1nbEwM/edit?usp=sharing
