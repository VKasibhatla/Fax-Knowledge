import preprocessing as preprocess
import tensorflow as tf
import LSTM

def run():
    pre = preprocess.parse()
    training_examples = pre[0][1][0]
    training_labels = pre[0][1][1]
    testing_examples = pre[1][1][0]
    testing_labels = pre[1][1][1]
    vocab = pre[2]
    knowledge_model = LSTM.LstmModel(num_labels=3,vocab_size = len(vocab))
    accuracy = test(knowledge_model,testing_examples,testing_labels)
    return accuracy

def train(model, train_input, train_labels):
    """
    Runs through all training examples.
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs of shape (num_inputs,)
    :param train_labels: train labels of shape (num_labels,3)
    :return: Average loss
    """
    total_loss = 0.0
    total_correct = 0.0
    total = 0.0
    total_square_error = 0.0
    batches = int(len(train_input) / model.batch_size)

    for batch in range(batches):
        with tf.GradientTape() as tape:
            inputs = train_input[ model.batch_size * batch : model.batch_size * (batch+1)]
            #print('shape',npinputs.shape)
            prbs = model.call(tf.convert_to_tensor(inputs), initial_state=None)
            labels = train_labels[model.batch_size * batch : model.batch_size * (batch+1)]
            loss = model.loss(prbs, labels)
            total_loss += loss

        guesses = tf.math.argmax(prbs, axis=1).numpy()
        for i in range(model.batch_size):
            print('batch',batch)
            #print('sentence',real_sentences[int(total)])
            #print('logits',prbs[i])
            print('guesses',guesses[i])
            total += 1
            total_square_error += (guesses[i] - labels[i]) ** 2
            print('labels[i]',labels[i])
            if labels[i]==guesses[i]:
                total_correct += 1
                print('CORRECT!')
            else:
                print('WRONG')
        #print('')
        print('accuracy',total_correct / total)
        print('total train',total)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss / batches

def test(model, test_inputs, test_labels):
    """
    Runs through all testing examples
    :param model: the LSTM model that will be used for testing 
    :param test_inputs: all tokenized sentences for testing
    :param test_labels: all labels of sentenes for testing
    :returns: accuracy of model predictions
    """

    pre = preprocess.parse()
    real_sentences = pre[1][0][0]
    real_labels = pre[1][0][1]

    total_correct = 0.0
    total = 0.0
    batches = int(len(test_inputs) / model.batch_size)

    for batch in range(batches):
        inputs = test_inputs[batch*model.batch_size:(batch+1)*model.batch_size]
        prbs = model.call(tf.convert_to_tensor(inputs), initial_state=None)
        guesses = tf.math.argmax(prbs, axis=1).numpy()

        labels = test_labels[batch*model.batch_size:(batch+1)*model.batch_size]

        for i in range(model.batch_size):
            print('sentence',real_sentences[int(total)])
            print('logits',prbs[i])
            print('label',real_labels[int(total)])
            total += 1
            if labels[i]==guesses[i]:
                total_correct += 1
                print('CORRECT')
            else:
                print('WRONG')
    print('accuracy',total_correct / total)
    print('total example',total)
    return total_correct / total


if __name__ == "__main__":
    print(run())