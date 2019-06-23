from flask import Flask, render_template, url_for, request, redirect

# import classifier.py
from classifier import predAll, pred

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
        choice = int(request.form['choice'])

        # category to print
        cat = ['identity hate', 'insult', 'obscene',
              'threat', 'toxic', 'severe toxic', 'negative']

        if choice == 6:
            pr0, pr1, pr2, pr3, pr4, pr5 = predAll(comment_text)
            # store values to a list
            allPr = [pr0, pr1, pr2, pr3, pr4, pr5]

            res = 0
            if 1 in allPr:
                res = 1

            return render_template('result.html', comment=comment_text,
                                   category=cat[choice], res=res)
        else:
            pr0, pa0 = pred(comment_text, choice)

            return render_template('result.html', comment=comment_text,
                                   category=cat[choice],
                                   res=pr0, a0=pa0)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/data")
def data():
    return render_template('data.html')

@app.route("/categories")
def categories():
    return render_template('categories.html')

if __name__ == '__main__':
    app.run(debug=True)
