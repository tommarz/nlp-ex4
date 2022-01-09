import tagger
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
dev_data = tagger.load_annotated_corpus('en-ud-dev.upos.tsv')

allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(train_data)

print('Vanilla LSTM')

params = tagger.get_best_performing_model_params()
params['input_rep'] = 0
blstm = tagger.initialize_rnn_model(params)
# tagger.train_rnn(blstm, train_data, verbose=1, bs=64)
PATH = 'model_input_rep_0.pt'
checkpoint = torch.load(PATH)
blstm['lstm'].load_state_dict(checkpoint)
blstm['lstm'] = blstm['lstm'].to(device)

print('BiLSTM+case')

params = tagger.get_best_performing_model_params()
cblstm = tagger.initialize_rnn_model(params)
# tagger.train_rnn(cblstm, train_data, verbose=1, bs=64)
PATH = 'model_input_rep_1.pt'
checkpoint = torch.load(PATH)
cblstm['lstm'].load_state_dict(checkpoint)
cblstm['lstm'] = cblstm['lstm'].to(device)

for model in ['baseline', 'hmm', 'blstm', 'cblstm']:
    sum_correct, sum_examples = 0, 0
    for test_sentence in dev_data:
        words_to_tag = [word for word, tag in test_sentence]
        if model == 'baseline':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'baseline': [perWordTagCounts, allTagCounts]})
        elif model == 'hmm':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'hmm': [A, B]})
        elif model == 'blstm':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'blstm': [blstm]})
        elif model == 'cblstm':
            tagged_sentence = tagger.tag_sentence(words_to_tag, {'cblstm': [cblstm]})

        correct, correctOOV, OOV = tagger.count_correct(test_sentence, tagged_sentence)
        sum_correct += correct
        sum_examples += len(words_to_tag)

    print(f"{model} - {sum_correct/sum_examples}")