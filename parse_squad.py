import json
import urllib
import pickle


def parse_squad(dataset):
    total_topics = 0
    total_questions = 0
    squad_json = []

    #every topic in the dataset
    for topic in dataset:
        total_topics += 1
        #every paragraph in the topic
        for passage in topic['paragraphs']:
            #every qas pairs in the passage
            for qas in passage['qas']:
                total_questions += 1
                text_question_pair = {}
                #title
                text_question_pair['topic'] = topic['title']
                #paragraph
                text_question_pair['paragraph'] = passage['context']
                #question
                text_question_pair['question'] = qas['question']
                #answers
                answer_start = 99999
                answer_end = -1
                for answer in qas['answers']:
                    answer_start = min(answer['answer_start'], answer_start)
                    answer_end = max(answer['answer_start'], answer_end)

                for idx in range(answer_end , len(text_question_pair['paragraph'])):
                    if text_question_pair['paragraph'][idx] == '?' or text_question_pair['paragraph'][idx] == '.' or \
                            text_question_pair['paragraph'][idx] == '!':
                        answer_end = idx-1
                        break
                    elif  idx == (len(text_question_pair['paragraph']) - 1):
                        answer_end = idx
                        break
                # for idx , letter in enumerate(text_question_pair['paragraph'][answer_end:]):
                #     if letter == '?' or letter == '.' or letter == '!' or letter ==',' or idx == len(text_question_pair['paragraph'])-1:
                #         answer_end = idx + answer_end
                #         break
                for idx in reversed(range(0 , answer_start)):
                    if text_question_pair['paragraph'][idx] == '?' or text_question_pair['paragraph'][idx] == '.' or \
                            text_question_pair['paragraph'][idx] == '!':
                        answer_start = idx+1
                        break
                    elif  idx == 0:
                        answer_start = idx
                        break
                # for idx, letter in enumerate(reversed(text_question_pair['paragraph'][:answer_start])):
                #     if letter == '?' or letter == '.' or letter == '!' or letter == ',' or idx + answer_start == 0:
                #         answer_start = idx + answer_start
                #         break

                answer_subSentences = text_question_pair['paragraph'][answer_start:answer_end + 1]

                text_question_pair['answer_subsentence'] = answer_subSentences
                squad_json.append(text_question_pair)
    print('Total topics' + str(total_topics))
    print('Total questions' + str(total_questions))
    return squad_json

#training
squad_train_filepath = 'data/train-v1.1.json'
save_path = 'data/parsed_train_data.json'
train_json = json.load(open(squad_train_filepath, 'r'))
train_dataset = train_json['data']
parsed_train_squad = parse_squad(train_dataset)
json.dump(parsed_train_squad, open(save_path, 'w'))

#Dev
save_path = 'data/parsed_dev_data.json'
squad_dev_filepath = 'data/dev-v1.1.json'
dev_json = json.load(open(squad_dev_filepath, 'r'))
dev_dataset = dev_json['data']
parsed_dev_squad = parse_squad(dev_dataset)
json.dump(parsed_dev_squad, open(save_path, 'w'))

#convert to pickle
train_filepath = 'data/parsed_train_data.json'
dev_filepath = 'data/parsed_dev_data.json'
train_set = json.load(open(train_filepath, 'r'))
dev_set = json.load(open(dev_filepath, 'r'))

train_paragraphs = []
train_subsent = []
train_questions = []
for section in train_set:
    train_paragraphs.append(section['paragraph'])
    train_questions.append(section['question'])
    train_subsent.append(section['answer_subsentence'])

dev_paragraphs = []
dev_subsent = []
dev_questions = []
for section in dev_set:
    dev_paragraphs.append(section['paragraph'])
    dev_questions.append(section['question'])
    dev_subsent.append(section['answer_subsentence'])

print(len(dev_paragraphs))
print(len(dev_questions))

assert len(train_paragraphs) == len(train_questions)
assert len(dev_paragraphs) == len(dev_questions)


def save_pickle(data, filename):
    """Saves the data into pickle format"""
    save_documents = open('data/' + filename + '.pickle', 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()


save_pickle(train_subsent, 'train_squad_paragraphs')
save_pickle(train_questions, 'train_squad_questions')
#save_pickle(train_subsent, 'train_squad_subsent')
save_pickle(dev_subsent, 'dev_squad_paragraphs')
save_pickle(dev_questions, 'dev_squad_questions')
#save_pickle(dev_subsent, 'dev_squad_subsent')
