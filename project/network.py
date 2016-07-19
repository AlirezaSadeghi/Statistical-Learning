from keras.layers import Dense, Input, Lambda
from keras.models import Sequential, Model
import keras.backend as K

doc_dim = 300  # 300 is dummy
hidden_dim = 128


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def create_base_network(input_dim):
    # TODO: replace with an RNN
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dense(hidden_dim, activation='relu'))
    return seq


def get_decoder_net(hidden_dim, out_dim):
    seq = Sequential()
    seq.add(Dense(out_dim, input_dim=hidden_dim))
    return seq


def non_sim_loss(y_true, y_pred):
    margin = hidden_dim
    return K.square(K.maximum(margin - y_pred, 0))


# assume number of related docs = 3 also add 1 irrelevant docs
main_in = Input((doc_dim,))
sim1_in = Input((doc_dim,))
sim2_in = Input((doc_dim,))
sim3_in = Input((doc_dim,))
nonsim1_in = Input((doc_dim,))

base_net = create_base_network(doc_dim)

topic_main = base_net(main_in)
topic_sim1 = base_net(sim1_in)
topic_sim2 = base_net(sim2_in)
topic_sim3 = base_net(sim3_in)
topic_nonsim1 = base_net(nonsim1_in)

dist1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
    [topic_main, topic_sim1])
dist2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
    [topic_main, topic_sim2])
dist3 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
    [topic_main, topic_sim3])
dist_non = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
    [topic_main, topic_nonsim1])

decoder = get_decoder_net(hidden_dim, doc_dim)
reconstruction = decoder(topic_main)

model = Model(input=[main_in, sim1_in, sim2_in, sim3_in, nonsim1_in],
              output=[reconstruction, dist1, dist2, dist3, dist_non])

model.compile('sgd', loss=['mse'] * 4 + [non_sim_loss])

# model.fit(xtrain, y_train)
