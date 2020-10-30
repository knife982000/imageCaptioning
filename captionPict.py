import argparse
from showAndTell import GRUAttention
from tensorflow.keras.models import load_model
import pickle
import math
import numpy as np
from preprocess_data import load_image
import tensorflow as tf


def generate(i, pict, text, model, width, end_id):
    c = []
    p = []
    e = []
    preds = model.predict([pict, text])[0, i-1, :]
    for _ in range(width):
        m = np.argmax(preds)
        c.append(m)
        p.append(preds[m])
        e.append(m == end_id)
        preds[m] = 0
    return c, p, e


def beam(pict, model, tokenizer, width, maxlen):
    end_id = tokenizer.word_index['<end>']
    start = np.zeros((1, maxlen), dtype=np.int32)
    start[0, 0] = tokenizer.word_index['<start>']
    candidates = [start]
    ended = [False]
    probs = [[1]]
    pict = np.reshape(pict, (1, -1, pict.shape[-1]))
    for i in range(1, maxlen):
        n_candidates = []
        n_ended = []
        n_probs = []
        all_ended = True
        for c, e, p in zip(candidates, ended, probs):
            if e:
                n_candidates.append(c)
                n_ended.append(e)
                n_probs.append(p)
            else:
                all_ended = False
                nc, n_p, ne = generate(i, pict, c, model, width, end_id)
                for vnc, vnp, vne in zip(nc, n_p, ne):
                    n_c = c.copy()
                    n_c[0, i] = vnc
                    n_candidates.append(n_c)
                    n_ended.append(vne)
                    n_probs.append(list(p) + [vnp])
                pass
        if all_ended:
            break
        log_prob = {e: np.average([math.log(x) for x in p]) for e, p in enumerate(n_probs)}
        index = list(range(len(n_probs)))
        index.sort(key=lambda x: log_prob[x], reverse=True)
        index = index[:width]
        candidates = [n_candidates[i] for i in index]
        probs = [n_probs[i] for i in index]
        ended = [n_ended[i] for i in index]
    return [c[0, :] for c in candidates], probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Caption generation')
    parser.add_argument('--model', '-m', type=str, help='model file path')
    parser.add_argument('--beam', '-b', type=int, default=3, help='beam search width')
    parser.add_argument('--len', '-l', type=int, default=20, help='text maxlen')
    parser.add_argument('--tokenizer', '-t', type=str, default='tokenizer.pickle', help='tokenizer file path')
    parser.add_argument('--pics', '-p', type=str, nargs='+', help='picts file paths')

    args = parser.parse_args()
    print(args)

    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    pict = []
    for p in args.pics:
        pict.append(load_image(p)[0].numpy())
    pict = np.asarray(pict)
    pict = model.predict(pict)

    del model
    with open(args.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    tf.keras.backend.clear_session()
    model = load_model(args.model, custom_objects={'GRUAttention': GRUAttention})
    for path, p in zip(args.pics, pict):
        print('*' * 80)
        print(path)
        cands, probs = beam(p, model, tokenizer, args.beam, args.len)
        for c, p in zip(cands, probs):
            print('\t' + ' '.join([tokenizer.index_word[w] for w in c]))
            print('\t{}'.format(p))
