import re
import numpy as np
from nltk.translate import bleu_score as nltkbleu
from collections import Counter
from nltk.util import ngrams

from generator import Generator

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


#BLUE,f1,distの計算の仕方はCR-Walker関数参考　
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
#    def remove_articles(text):
#        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    #return white_space_fix(remove_articles(remove_punc(lower(s))))
    return white_space_fix(remove_punc(lower(s)))

def bleu(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    try:
        return nltkbleu.sentence_bleu(
            [normalize_answer(a).split(" ") for a in answers],
            normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method7,
        )
    except:
        return 0

def prec_recall_f1_score(pred_items, gold_items):
    """
    Computes precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)

def generate_n_grams(x, n):
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    # print(x, n_grams)
    # for n_gram in n_grams:
    #     x.append(' '.join(n_gram))
    return n_grams


def distinct_n_grams(tokenized_lines, n):

    n_grams_all = set()
    for line in tokenized_lines:
        n_grams = generate_n_grams(line, n)
        # print(line, n_grams)
        n_grams_all |= n_grams
    total_len=0
    for item in tokenized_lines:
        total_len+=len(item)

    return len(set(n_grams_all)), len(set(n_grams_all)) / total_len#len(tokenized_lines)

def evaluate_gen_redial(dataset): 
    #lines = [item['generated'].strip() for item in generated_utters]
    genr = Generator()

    bleu_array = []
    f1_array = []

    """
    print(f"dataset[0]:{dataset[0]}")
    dataset[0]:{'context': "Hi, did you see @196336 ? Yes it was a pretty good movie. Then you would like @114851 if you haven't seen it. You like Sci Fi stuff like @204292 ? I am more of a @143189 kind of person. Did you see any of the Star Wars parody movies? @143189 was pretty funny? I do not think I have seen that one. Who is in it? Have you seen any cartoon movies like @204322 ? Yes those are really funny. I think I will watch @143189. Thanks for the recommendation. @143189 r Wars Clones is a claymation movie. That sounds great. It is. You like blood and gore movies? If so, you might like @128905 . [SEP] No I do not like that genre of movie. The star wars movie is perfect for me<|endoftext|>", 'utterance': 'No I do not like that genre of movie. The star wars movie is perfect for me', 'mentioned': [5004, 752, 5413, 30448, 2208, 30450, 30438, 5414, 30436, 1487], 'node_candidate1': [5004, 752, 5413, 30448, 2208, 30450, 30438, 5414, 30436, 1487], 'label_1': [5], 'node_candidate2': [[]], 'label_2': [[]], 'intent': 'chat', 'new_mentioned': [30450, 30455], 'dialog_num': 1, 'system_turn': 5, 'label_rec': [5], 'context_org_addSEP': "Hi, did you see @196336 ? Yes it was a pretty good movie. Then you would like @114851 if you haven't seen it. You like Sci Fi stuff like @204292 ? I am more of a @143189 kind of person. Did you see any of the Star Wars parody movies? @143189 was pretty funny? I do not think I have seen that one. Who is in it? Have you seen any cartoon movies like @204322 ? Yes those are really funny. I think I will watch @143189. Thanks for the recommendation. @143189 r Wars Clones is a claymation movie. That sounds great. It is. You like blood and gore movies? If so, you might like @128905 . [SEP] "}
    """
    prompt = dataset[0]["context_org_addSEP"]
    print(f"eval_prompt:{prompt}")
    print()

    print(f"test prompt")
    prompt = "what is your favorite movie?"
    print(f"eval_prompt:{prompt}")
    print()
    
    generated  = genr.generate(prompt)
    print(f"eval_generated:{generated}")
    print()
    ground = dataset[0]['utterance']
    print(f"eval_ground:{ground}")
    print()

    #Blueの測り方これであっているのか？
    bleu_array.append(bleu(generated, ground))
    f1_array.append(f1_score(generated, ground))

    # #カバレッジをここに足したい
    # #cov = len(tot_rec_item)/6924
    # tot_rec_item = list(dict.fromkeys(tot_rec_item))
    # #tot_rec_item_hu = list(dict.fromkeys(tot_rec_item_hu))
    # cov = len(tot_rec_item)/6924
    # #cov_hu = len(tot_rec_item_hu)/6924
    # #print(f"coverage:{cov}")
    #print(f"coverage_human:{cov_hu}")

    Bleu=np.mean(bleu_array)
    f1=np.mean(f1_array)
    print("BLEU:",Bleu)
    print("F1:",f1)


    # #distってサブワード分割単位でいいんだろうか
    # dist=[]


    # tokenized = [line.split() for line in lines]
    # for n in range(1, 6):
    #     cnt, percent = distinct_n_grams(tokenized, n)
    #     dist.append(percent)
    #     #print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')
    # print(f"Bleu:{Bleu}")
    # print(f"f1:{f1}")
    # print(f"dist:{dist}")

    #return Bleu, f1, dist

    return Bleu, f1




# if __name__ == '__main__':
#     genr = Generator()
#     gen_text = genr.generate("What your favorite movie?")
#     print(f"gen_text:{gen_text}")

