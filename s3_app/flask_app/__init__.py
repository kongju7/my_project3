from flask import Flask, render_template, request
import re
from konlpy.tag import Okt
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def main_get(): 
        review = None 
        result = None
        return render_template('search.html', result = result)

    @app.route('/result', methods = ['POST'])
    def sentiment_predict():
        review = request.form.get('review', False)

        MODEL_PATH = '/data/shop_models'
        loaded_model = load_model(MODEL_PATH)

        okt = Okt()
        tokenizer = Tokenizer()

        DATA_PATH = '/data/'
        DATA_CONFIGS = 'data_configs.json'
        prepro_configs = json.load(open(DATA_PATH+DATA_CONFIGS,'r'))
        word_vocab = prepro_configs['vocab']

        tokenizer.fit_on_texts(word_vocab)
        stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', 
                    '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
        max_len = 80
        review = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', review)
        review = okt.morphs(review)
        review = [word for word in review if not word in stopwords]
        encoded = tokenizer.texts_to_sequences([review])
        pad_new = pad_sequences(encoded, maxlen = max_len)

        score = float(loaded_model.predict(pad_new))
        if (score > 0.5):
            result = "고객님의 리뷰를 긍정 리뷰로 분류할 수 있는 확률은 {:.2f}%입니다.".format(score * 100)
        else:
            result = "고객님의 리뷰를 부정 리뷰로 분류할 수 있는 확률은 {:.2f}%입니다.".format((1 - score) * 100)

        return render_template('search.html', result = result)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug = True)
