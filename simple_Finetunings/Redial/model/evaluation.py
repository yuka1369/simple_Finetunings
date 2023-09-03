import re
import numpy as np
from nltk.translate import bleu_score as nltkbleu
from collections import Counter
from nltk.util import ngrams
from generator import Generator

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


#BLUEの計算はCR-Walker関数参考　
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

def evaluate_gen_redial(dataset): 
    #lines = [item['generated'].strip() for item in generated_utters]
    genr = Generator()

    bleu_array = []
    f1_array = []

    print(f"dataset[0]:{dataset[0]}")
    prompt = dataset["context"]
    generated  = genr.generate(prompt)
    ground = dataset["uttaranunce"]

    #Blueの測り方これであっているのか？
    bleu_array.append(bleu(generate, ground))
    f1_array.append(f1_score(generate, ground))

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
    dist=[]
    #print("BLEU:",Bleu)
    #print("F1:",f1)


    tokenized = [line.split() for line in lines]
    for n in range(1, 6):
        cnt, percent = distinct_n_grams(tokenized, n)
        dist.append(percent)
        #print(f'Distinct {n}-grams (cnt, percentage) = ({cnt}, {percent:.3f})')
    
    return Bleu, f1, dist

# if __name__ == '__main__':
#     genr = Generator()
#     gen_text = genr.generate("What your favorite movie?")
#     print(f"gen_text:{gen_text}")

