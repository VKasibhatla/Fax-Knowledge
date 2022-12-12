import preprocessing as preprocess
import tensorflow as tf
import numpy as np
from paper_model import FaxModel

def train(model, train_input, train_labels, batch_size):
    """
    Runs through all training examples.
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs of shape (num_inputs,)
    :param train_labels: train labels of shape (num_labels,3)
    :return: Average loss
    """
    batches = int(len(train_input) / batch_size)
    total_loss = 0
    total_seen = total_correct = 0
    for batch in range(batches):
        with tf.GradientTape() as tape:
            inputs = train_input[batch*batch_size:(batch+1)*batch_size]
        
            prbs = model.call(inputs)
            labels = train_labels[batch*batch_size:(batch+1)*batch_size]
            guesses =tf.math.argmax(prbs, axis=1).numpy() 
        
            loss = model.loss(prbs, labels)
            labels = tf.math.argmax(labels, axis=1).numpy()
            total_loss += loss

            for i in range(batch_size):
                total_seen += 1
                if guesses[i]==labels[i]:
                    total_correct += 1
            
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("finishied batch" + str(batch))
        print("accuracy " + str(total_correct/ total_seen))

    return {"loss" : total_loss/ batches, "acc": total_correct/ total_seen}

def test(model, test_input, test_labels, batch_size):
    """
    Runs through all testing examples.
    :param model: the initilized model
    :param train_inputs: train inputs of shape (num_inputs,)
    :param train_labels: train labels of shape (num_labels, 3)
    :return: Average loss
    """
    batches = int(len(test_input) / batch_size)
    total_seen = total_correct = 0

    for batch in range(batches):
            with tf.GradientTape() as tape:
                inputs = test_input[batch*batch_size:(batch+1)*batch_size]
                output = model.call(inputs)
                guesses = tf.math.argmax(output, axis=1).numpy()
                labels = test_labels[batch*batch_size:(batch+1)*batch_size]
                labels = tf.math.argmax(labels, axis=1).numpy()

                for i in range(batch_size):
                    total_seen += 1
                    if guesses[i]==labels[i]:
                        total_correct += 1
            print(inputs)
            print(guesses)
            print(labels)
            print("finishied batch" + str(batch))
            print("accuracy " + str(total_correct/ total_seen))

    return {"acc": total_correct / total_seen}


if __name__ == "__main__":
    """
    Read in COVID rumors dataset and initialize/train/test model.
    """ 

    # demo_sentences = ["Australians trapped in Wuhan say they need to pay $673 to be rescued.", 
    # "You will get a free coronavirus test by donating blood.",
    # "The 2019 coronavirus causes sudden death syndrome.",
    # "COVID-19 emerged in December 2019."]

    demo_sentences = []
    demo_labels = [1,0,0,2]

    training = False
    paper_model = True

    ## Read in COVID rumors data,
    pre = preprocess.parse()
    
    model = FaxModel(3) if paper_model else None

    if(training):
        if(paper_model):
            train_inputs = pre[0][0][0]
            train_labels = pre[0][0][1]
            test_inputs = pre[1][0][0]
            test_labels = pre[1][0][1]
            train(model, train_inputs, train_labels, 100)
            
            

    if (len(demo_sentences) == 0):
        test_metrics = test(model,test_inputs, test_labels, 1)
        print('Testing Performance:', test_metrics)
    else:
        if(paper_model):
            for i in range(len(demo_sentences)):
                prbs = model.call(demo_sentences[i:i+1])
                guesses = tf.math.argmax(prbs, axis=1).numpy()
                print('Sentence:', demo_sentences[i])
                print('True Label:', demo_labels[i])
                print('Predicted Label:', guesses[0])
                print('----------')

