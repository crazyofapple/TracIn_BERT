import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense, Activation
from keras.initializers import RandomNormal
from keras.regularizers import l2
import tensorflow as tf
from copy import deepcopy
set_gelu('tanh') 

maxlen = 128
batch_size = 2048
config_path = '/raid/ldf/wanganqi/bert_pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/raid/ldf/wanganqi/bert_pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/raid/ldf/wanganqi/bert_pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'
# tf.enable_eager_execution()

# tf.executing_eagerly() 

class Model:
    @staticmethod
    def loadmodel(path):
        return loadmodel(path)

    def ___init__(self, path):
       self.model = self.loadmodel(path)
       self.graph = tf.get_default_graph()

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)

def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return l2(
        l2_weight_decay) if use_l2_regularizer else None
index_to_classname = {"0":["0", "not match"], "1": ["1", "match"]}
def load_data(filename, number=None):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for ind, l in enumerate(f):
            if ind == 0:
                continue
            if number is not None and ind == number:
                break
            text1, text2, label = l.strip().split('\t')
            D.append((text1, text2, int(label)))
    return D


# 加载数据集
train_data = load_data('../lcqmc/train.tsv', number=50000)
valid_data = load_data('../lcqmc/dev.tsv', number=1)
test_data = load_data('../lcqmc/test.tsv', number=50)
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

CHECKPOINTS_PATH_FORMAT = "model_epoch_{}_50k_SGD.weights"
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

# output = Dropout(rate=0)(bert.model.output)
# output = Dense(
#     units=2, activation='softmax', kernel_initializer=bert.initializer
# )(output)
x = Dense(
    2,
    kernel_initializer=RandomNormal(stddev=0.01),
    kernel_regularizer=_gen_l2_regularizer(),
    bias_regularizer=_gen_l2_regularizer(),
    name='fc2')(
        bert.model.output)

# A softmax that is followed by the model loss cannot be done
# in float16 due to numeric issues. So we pass dtype=float32.
output = Activation('softmax', dtype='float32')(x)
models_penultimate = []
models_last = []
sessions = []

def run(inputs):
    input_ids_and_segment_ids, labels = inputs
    # ignore bias for simplicity
    loss_grads = []
    activations = []
    def top_k(input, k=1, sorted=True):
        """Top k max pooling
        Args:
            input(ndarray): convolutional feature in heigh x width x channel format
            k(int): if k==1, it is equal to normal max pooling
            sorted(bool): whether to return the array sorted by channel value
        Returns:
            ndarray: k x (height x width)
            ndarray: k
        """
        ind = np.argpartition(input, -k)[..., -k:]
        def get_entries(input, ind, sorted):
            if len(ind.shape) == 1:
                if sorted:
                    ind = ind[np.argsort(-input[ind])]
                return input[ind], ind
            output, ind = zip(*[get_entries(inp, id, sorted) for inp, id in zip(input, ind)])
            return np.array(output), np.array(ind)
        return get_entries(input, ind, sorted)
    for mp, ml, sess in zip(models_penultimate, models_last, sessions):
        with sess.as_default():
            # h = mp(tf.convert_to_tensor(input_ids_and_segment_ids))
            h = mp.predict(input_ids_and_segment_ids)
            # print(K.eval(h))
            # print(h)
            logits = ml(tf.convert_to_tensor(h))
            probs = tf.nn.softmax(logits)
            # probs = logits
            
            loss_grad = tf.one_hot(labels, 2) - probs
            # print(K.eval(tf.one_hot(labels, 2)))
            activations.append(h)
            # print(h.shape)
            # print((h))
            # print(K.eval(loss_grad))
            # print("#"*50)
            loss_grads.append(K.eval(loss_grad))
            probs = K.eval(probs)
    # Using probs from last checkpoint
    probs, predicted_labels = top_k(probs, k=1)
    # exit(0)
    return np.stack(loss_grads, axis=-1), np.stack(activations, axis=-1), labels, probs, predicted_labels


# epoch#1 #train_loss 0.2231 #train_acc 0.9069 val_acc: 0.86242, best_val_acc: 0.86242, test_acc: 0.84256v
# epoch#2 #train_loss 0.1650 #train_acc 0.9343 val_acc: 0.86980, best_val_acc: 0.86980, test_acc: 0.83688
# epoch#3 #train_loss 0.1253 #train_acc 0.9519 val_acc: 0.87139, best_val_acc: 0.87139, test_acc: 0.86552
# epoch#4 #train_loss 0.0964 #train_acc 0.9635 val_acc: 0.88412, best_val_acc: 0.88412, test_acc: 0.86768
# epoch#5 #train_loss 0.0770 #train_acc 0.9713 val_acc: 0.87412, best_val_acc: 0.88412, test_acc: 0.85632
for i in range(len([0,2])):
    sessions.append(tf.Session())

checkpoints = ["model_step_100_SGD.weights", "model_epoch_5_50k_SGD.weights"]
for ind, i in enumerate([0,2]):
    
    with sessions[ind].as_default():
        model = keras.models.Model(bert.model.input, output)
        # model.load_weights(CHECKPOINTS_PATH_FORMAT.format(i))
        model.load_weights(checkpoints[ind])
        # model.summary()
        models_penultimate.append(keras.models.Model((model.layers[0].input, model.layers[1].input), model.layers[-3].output))
        models_last.append(model.layers[-2])
        # sessions.append(sess)

# for x_true, y_true in test_generator:
#     # y_pred = model.predict(x_true).argmax(axis=1)
#     loss_grads, activations, labels, probs, predicted_labels = run((x_true, y_true))
#     # print(loss_grads)
#     print(activations)
#     break

def get_trackin_grad(ds):
    loss_grads_np = []
    activations_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []
    for x_true, y_true in ds:
        loss_grads_replica, activations_replica, labels_replica, probs_replica, predictied_labels_replica = run((x_true, y_true))
        for loss_grads, activations, labels, probs, predicted_labels in zip(
            loss_grads_replica,
            activations_replica, 
            labels_replica, 
            probs_replica, 
            predictied_labels_replica):
            loss_grads_np.append(loss_grads)
            activations_np.append(activations)
            labels_np.append(labels)
            probs_np.append(probs)
            predicted_labels_np.append(predicted_labels)
    return {
            'loss_grads': np.concatenate(loss_grads_np),
            'activations': np.concatenate(activations_np),
            'labels': np.concatenate(labels_np),
            'probs': np.concatenate(probs_np),
            'predicted_labels': np.concatenate(predicted_labels_np)
            }    
trackin_train = get_trackin_grad(train_generator)

def find(loss_grad=None, activation=None, topk=50):
    if loss_grad is None and activation is None:
        raise ValueError('loss grad and activation cannot both be None.')
    scores = []
    scores_lg = []
    scores_a = []
    for i in range(len(trackin_train['labels'])):
        if loss_grad is not None and activation is not None:
            lg_sim = np.sum(trackin_train['loss_grads'][i] * loss_grad)
            a_sim = np.sum(trackin_train['activations'][i] * activation)
            scores.append(lg_sim * a_sim)
            scores_lg.append(lg_sim)
            scores_a.append(a_sim)
        elif loss_grad is not None:
            scores.append(np.sum(trackin_train['loss_grads'][i] * loss_grad))
        elif activation is not None:
            scores.append(np.sum(trackin_train['activations'][i] * activation))    

    opponents = []
    proponents = []
    indices = np.argsort(scores)
    for i in range(topk):
        index = indices[-i-1]
    
        proponents.append((
            index,
            trackin_train['probs'][index],
            index_to_classname[str(trackin_train['predicted_labels'][index])][1],
            index_to_classname[str(trackin_train['labels'][index])][1], 
            scores[index],
            scores_lg[index] if scores_lg else None,
            scores_a[index] if scores_a else None))
        index = indices[i]
        opponents.append((
            index,
            trackin_train['probs'][index],
            index_to_classname[str(trackin_train['predicted_labels'][index])][1],
            index_to_classname[str(trackin_train['labels'][index])][1],
            scores[index],
            scores_lg[index] if scores_lg else None,
            scores_a[index] if scores_a else None))  
    return opponents, proponents


def find_and_show(trackin_dict, idx, vector='influence'):
    if vector == 'influence':
        op, pp = find(trackin_dict['loss_grads'][idx], trackin_dict['activations'][idx])

    print('Query example from validation: ')
    print('label: {}, prob: {}, predicted_label: {}'.format(
        index_to_classname[str(trackin_dict['labels'][idx])][1], 
        trackin_dict['probs'][idx], 
        index_to_classname[str(trackin_dict['predicted_labels'][idx])][1]))
        
    this_data = test_data[idx]
    print(this_data)
    print("="*50)  
    print('Proponents: ')
    for p in pp:
        print('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(p[3], p[1], p[2], p[4]))
        if p[5] and p[6]:
            print('error_similarity: {}, encoding_similarity: {}'.format(p[5], p[6]))
        p_data = train_data[p[0]]
        print(p_data)
    print("="*50)
    print('Opponents: ')
    for o in op:
        print('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(o[3], o[1], o[2], o[4]))
        if o[5] and o[6]:
            print('error_similarity: {}, encoding_similarity: {}'.format(o[5], o[6]))
        o_data = train_data[o[0]]
        print(o_data)

    print("="*50) 
    
trackin_val = get_trackin_grad(test_generator)

find_and_show(trackin_val, 1, 'influence')

# find_and_show(trackin_val, 8, 'influence') 

# find_and_show(trackin_val, 21, 'influence')

# find_and_show(trackin_val, 37, 'influence')

find_and_show(trackin_val, 39, 'influence')
