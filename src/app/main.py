import pandas as pd
import pickle
import os

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

from sklearn.linear_model import LinearRegression
from textblob import TextBlob

# importando modelo
modelo = pickle.load(open('../../models/modelo.sav', 'rb'))

colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

# definindo a rota que os usuarios/devs vao acessar
# dentro de () a gente passa a rota
# a rota base que normalmente eh a home, vamos deixar com a / pois o usuario vai acessar com o
# endereco base
@app.route('/')
# definindo a funcao a ser executada quando chegar na rota
def home():
    return "Minha primeira API."

#-----------------------------------------------------------------------------#

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    
    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity

    return "polaridade: {}".format(polaridade)

#-----------------------------------------------------------------------------#

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():

    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])

    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')