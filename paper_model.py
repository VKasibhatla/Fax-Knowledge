import spacy
import numpy as np
from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf
from allennlp.predictors.predictor import Predictor
import copy

nlp = spacy.load("en_core_web_md")
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
        "textual_entailment")
entailment = predictor
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = TFBertForMaskedLM.from_pretrained('bert-base-cased')

class FaxModel(tf.keras.Model):
    '''
    Model based on the paper https://aclanthology.org/2020.fever-1.5.pdf


    1. Masked the last named entity found by the NER model.
    2. Get the top-1 predicted token from the LM, and fill in the [MASK] accordingly to create the “evidence” sentence
    3. Using the claim and generated “evidence” sen- tence, obtain entailment “features” using out-puts from the last layer of the pretrained entailment model (before the softmax).
    4. Input the entailment features into a multi-layer perceptron for final fact-verification prediction.
    
    '''
    def __init__(self, num_labels):
        super(FaxModel, self,).__init__()
    
        self.hidden_layer = tf.keras.layers.Dense(100)
        self.dense = tf.keras.layers.Dense(num_labels, activation='softmax')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)

    def call(self, inputs):
        """
        Performs a forward pass on given `inputs` which contains a list of sentences

        :param inputs: a list of sentences
        :return: the batch item probabilities
        """
        get_logits = lambda x: self.get_entailment_features(x)
        whole_output = tf.stack([get_logits(sentence) for sentence in inputs])
        whole_output = tf.reshape(whole_output,shape=(len(inputs),3))
        output = self.hidden_layer(whole_output)
        output = self.dense(output)
        return output

    def get_entailment_features(self, sentence):
        '''
        Obtain entailment “features” using outputs from the last layer of the pretrained entailment model
        
        :param sentence: a given sentence
        :return: logits representing entailment features
        '''
    
        new_str = copy.deepcopy(sentence)
        doc = nlp(new_str)
        
        if(len(list(doc.noun_chunks)) == 0) :
            new_str = new_str[:-1]
            words =  new_str.split()
            words[-1] = "[MASK]"
            new_str = ' '.join(words)
            new_str += '.'
        else:
            new_str =  new_str.replace(list(doc.noun_chunks)[-1].text, "[MASK]")
        new_str =  new_str.replace("[MASK]", self.get_prediction(new_str))
        logits = entailment.predict(sentence, new_str)["logits"]
        return logits

    def get_prediction(self, input_string) -> str:
        '''
        Get the top-1 predicted token from the LM, and fill in the [MASK] accordingly to create the “evidence” sentence
        
        :param input_string: a given sentence with a masked noun
        :return: the predicted sentence
        '''

        tokenized_inputs = bert_tokenizer(input_string, return_tensors="tf")
        outputs = bert_model(tokenized_inputs["input_ids"])

        top_k_indices = tf.math.top_k(outputs.logits, 1).indices[0].numpy()
        decoded_output = bert_tokenizer.batch_decode(top_k_indices)
        mask_token = bert_tokenizer.encode(bert_tokenizer.mask_token)[1:-1]
        mask_index = np.where(tokenized_inputs['input_ids'].numpy()[0]==mask_token)[0][0]

        decoded_output_words = decoded_output[mask_index]

        return decoded_output_words

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param probs: a matrix containing probabilities
        :param labels: matrix containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels,probs))
