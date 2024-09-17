### BERT based Sentiment Analyzer
To run: python2 main.py
download 417.7MB model from the link into the main folder : https://drive.google.com/file/d/1mfv2rwl3l2k0qV8dD9tDfWiCirCRD546/view?usp=sharing


--To build example

docker build -t drphilipobiorah/bert_sentiment_analyzer:latest .    


---To run docker example

docker container run -d -p 5000:5000 drphilipobiorah/bert_sentiment_analyzer:latest 



---build for another architecture
docker build --platform linux/ppc64le -t drphilipobiorah/bert_sentiment_analyzer .
