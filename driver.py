import tagger
from collections import defaultdict
import torch

if __name__ == '__main__':
    train = tagger.load_annotated_corpus('en-ud-train.upos.tsv')
    test = tagger.load_annotated_corpus('en-ud-dev.upos.tsv')

    allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B = tagger.learn_params(train)

    model_dict = {'embedding_dimension': 100, 'hidden_dim': 256, 'max_vocab_size': 5000, 'num_of_layers': 1,
                  'output_dimension': len(allTagCounts), 'min_frequency': 10, 'input_rep': 1,
                  'pretrained_embeddings_fn': 'glove.6B.100d.txt', 'data_fn': 'en-ud-train.upos.tsv', 'dropout': 0.0}

    # model_dict = tagger.get_best_performing_model_params()

    # model_dict['input_rep'] = 0

    model = tagger.initialize_rnn_model(model_dict)

    # PATH = 'model2.pt'
    #
    # checkpoint = torch.load(PATH)
    # model['lstm'].load_state_dict(checkpoint['model_state_dict'])

    tagger.train_rnn(model, train)

    models = {'baseline': [perWordTagCounts, allTagCounts], 'hmm': [A, B], 'blstm': [model]}
    accuracy_dict = {k: defaultdict(int) for k in models}
    for i in range(len(test)):
        for name, model in models.items():
            raw_sentence = test[i]
            sentence = [x[0] for x in raw_sentence]

            tagged_sentence = tagger.tag_sentence(sentence, {name: model})
            correct, correctOOV, OOV = tagger.count_correct(raw_sentence, tagged_sentence)
            accuracy = round(correct / len(sentence), 3)
            accuracy_dict[name][i] = accuracy
    for k in accuracy_dict:
        accuracy_dict[k] = round(sum(accuracy_dict[k].values()) / len(test), 3)
    print(f'Total accuracy: {accuracy_dict}')
