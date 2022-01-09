from tagger import *
from collections import defaultdict
import torch

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    use_seed()

    train = load_annotated_corpus('en-ud-train.upos.tsv')
    test = load_annotated_corpus('en-ud-dev.upos.tsv')

    allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = learn_params(train)

    model_dict = get_best_performing_model_params()

    print('BiLSTM+case')

    cblstm = initialize_rnn_model(model_dict)

    # train_rnn(cblstm, train, verbose=1, bs=128)
    # Load pretrained model

    PATH = 'model_input_rep_1.pt'
    checkpoint = torch.load(PATH)
    cblstm['lstm'].load_state_dict(checkpoint)
    cblstm['lstm'] = cblstm['lstm'].to(device)

    # Train Vanilla BiLSTM

    model_dict['input_rep'] = 0

    blstm = initialize_rnn_model(model_dict)

    # Train BiLSTM+case

    # train_rnn(blstm, train, verbose=1, bs=128)

    PATH = 'model_input_rep_0.pt'
    checkpoint = torch.load(PATH)
    blstm['lstm'].load_state_dict(checkpoint)
    blstm['lstm'] = blstm['lstm'].to(device)

    models = {'baseline': [perWordTagCounts, allTagCounts], 'hmm': [A, B], 'blstm': [blstm], 'cblstm': [cblstm]}
    accuracy_dict = defaultdict(int)
    for name, model in models.items():
        total_correct = 0
        total_words = 0
        for raw_sentence in test:
            sentence = [x[0] for x in raw_sentence]
            tagged_sentence = tag_sentence(sentence, {name: model})
            correct, correctOOV, OOV = count_correct(raw_sentence, tagged_sentence)
            total_correct += correct
            total_words += len(sentence)
        print(f'{name} : {round(total_correct / total_words, 3)}')
