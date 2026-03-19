import collections
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import rnn as rnn_lstm

start_token = 'G'
end_token = 'E'


def process_poems(file_name):
    """
    Returns:
      poems_vector: list[list[int]]
      word_int_map: dict[str,int]
      vocabularies: tuple[str]
    """
    poems = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if ':' in line:
                    _, content = line.split(':', 1)
                elif '：' in line:
                    _, content = line.split('：', 1)
                else:
                    content = line

                content = content.replace(' ', '')
                if any(bad in content for bad in ['_', '(', ')', '[', ']', '（', '）', '《', '》']):
                    continue
                if start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                poems.append(start_token + content + end_token)
            except ValueError:
                continue

    poems = sorted(poems, key=lambda line: len(line))

    all_words = []
    for poem in poems:
        all_words.extend(list(poem))

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    vocabularies = words[:len(words)] + (' ',)
    word_int_map = dict(zip(vocabularies, range(len(vocabularies))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, vocabularies


def generate_batch(batch_size, poems_vec):
    n_chunk = len(poems_vec) // batch_size
    x_batches, y_batches = [], []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y = row[1:] + [row[-1]]
            y_data.append(y)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training(poems_file='./poems.txt', model_path='./poem_generator_rnn', epochs=30, batch_size=100):
    poems_vector, word_to_int, _ = process_poems(poems_file)
    print('finish loading data, poems:', len(poems_vector), 'vocab:', len(word_to_int) + 1)

    torch.manual_seed(5)
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=batch_size,
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    )

    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss()

    for epoch in range(epochs):
        batches_inputs, batches_outputs = generate_batch(batch_size, poems_vector)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]

            loss = 0.0
            for index in range(batch_size):
                x = np.array(batch_x[index], dtype=np.int64)
                y = np.array(batch_y[index], dtype=np.int64)

                x = Variable(torch.from_numpy(np.expand_dims(x, axis=1)))
                y = Variable(torch.from_numpy(y))

                pre = rnn_model(x)
                loss = loss + loss_fun(pre, y)

                if index == 0:
                    _, pred = torch.max(pre, dim=1)
                    print('prediction', pred.data.tolist())
                    print('b_y       ', y.data.tolist())
                    print('*' * 30)

            loss = loss / batch_size
            print('epoch', epoch, 'batch', batch, 'loss:', float(loss.data.tolist()))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1.0)
            optimizer.step()

            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), model_path)
                print('finish save model')

    torch.save(rnn_model.state_dict(), model_path)
    print('training done, model saved to', model_path)


def to_word(predict, vocabs):
    sample = int(np.argmax(predict))
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):
    chars = []
    for w in poem:
        if w == start_token or w == end_token:
            continue
        chars.append(w)
    text = ''.join(chars)
    if '。' in text:
        for s in text.split('。'):
            if s:
                print(s + '。')
    else:
        print(text)


def gen_poem(begin_word, poems_file='./poems.txt', model_path='./poem_generator_rnn', max_len=60):
    poems_vector, word_int_map, vocabularies = process_poems(poems_file)
    _ = poems_vector  # keep for compatibility/debug

    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=64,
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    )

    rnn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    rnn_model.eval()

    if begin_word not in word_int_map:
        # Fallback to a common known token in vocab.
        if '日' in word_int_map:
            begin_word = '日'
        else:
            begin_word = next((w for w in vocabularies if w not in [start_token, end_token, ' ']), vocabularies[0])

    # Prefix with start token to match training data format.
    poem = start_token + begin_word
    word = begin_word

    while word != end_token:
        inp_ids = [word_int_map[w] for w in poem if w in word_int_map]
        if len(inp_ids) == 0:
            inp_ids = [word_int_map[start_token]]
        inp = np.array(inp_ids, dtype=np.int64)
        inp = Variable(torch.from_numpy(inp))
        output = rnn_model(inp, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        if len(poem) > max_len + 1:  # +1 for start token
            break

    return poem


if __name__ == '__main__':
    # Set True when you want to train. Training is slow.
    TRAIN = False

    base_dir = os.path.dirname(__file__)
    poems_file = os.path.join(base_dir, 'poems.txt')
    model_path = os.path.join(base_dir, 'poem_generator_rnn')

    if TRAIN:
        run_training(poems_file=poems_file, model_path=model_path, epochs=30, batch_size=100)

    if not os.path.exists(model_path):
        print('Model file not found:', model_path)
        print('Set TRAIN=True and run once to train the model first.')
    else:
        for bw in ['日', '红', '山', '夜', '湖', '海', '月']:
            print('\n[begin word]', bw)
            pretty_print_poem(gen_poem(bw, poems_file=poems_file, model_path=model_path))
