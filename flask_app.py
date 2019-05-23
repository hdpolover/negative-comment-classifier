from flask import Flask, render_template, url_for, request, redirect

#import classifier.py
from classifier import pred

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/clf")
def clf():
    return render_template("classifier.html")

@app.route("/predi", methods=['POST'])
def predi():
    if request.method == 'POST':
        comment_text = request.form['txt']
        choice = request.form['choice']

        #category to print
        cat = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate', 'negative']
        #cast to int
        index = int(choice)

        if choice == "6":
            pr0, pr1, pr2, pr3, pr4, pr5,
            pa0, pa1, pa2, pa3, pa4, pa5 = pred(comment_text, choice)

            return render_template('result.html', comment = comment_text, category = cat[index],
            pr0 = pr0, pr1 = pr1, pr2 = pr2, pr3 = pr3, pr4 = pr4, pr5 = pr5, 
            pa0 = pa0, pa1 = pa1, pa2 = pa2, pa3 = pa3, pa4 = pa4, pa5 = pa5)
        else:
            pr0, pa0 = pred(comment_text, choice)

            return render_template('result.html', comment = comment_text, category = cat[index],
            pr0 = pr0, pa0 = pa0)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/view")
def view():
    return render_template('view.html')

if __name__ == '__main__':
	app.run(debug=True)
