# Fax-Knowledge

Group members:

* Josue Cruz (jcruz14)
* Varun Kasibhatla (vkasibha)

This project is a reimplementation of the work in [Language Models as Fact Checkers?](https://aclanthology.org/2020.fever-1.5.pdf) (2020).

The contents of the data.txt file was downloaded from https://github.com/MickeysClubhouse/COVID-19-rumor-dataset/blob/master/Data/news/news.csv.


# Setup
Steps to install spaCy:
```
conda activate dl3
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```

Steps to install BERT Model
```
pip install transformers
```

Steps to install allenNLP Model
```
pip install allennlp
```
Recommended to install PyTorch from https://pytorch.org

# Running the Models

This project uses the conda environment found in the [CSCI1470 Setup Guide](https://docs.google.com/document/d/1Qcss983uPe25bn-gH4DBVGq6X1viyEHW/edit)

This project contains two models:
* FaxModel: Reimplementation of the model suggested in the paper
* KnowledgeModel: Plain LSTM Recurrent Neural Network Model

Change parameters based on needs in main function: If you would like to train a model change variable to true and select either FaxModel (paper_model = True) or KnowledgeModel(paper_model = False)
```
train_model = False
paper_model = True
```

Note: During implemenation, training for FaxModel was around 40 minutes.

To train, test or see demo results for a model from the paper, run:
```
python main.py
```
