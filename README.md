# sentiment-analysis
environment:

We start a small project to build a simple system to automatically give labels to comments.
If you want to use the web part, make sure you add xampp in your pc or some other apache and sql service.
we run the file under linux and win10, with python3.5.2 and tensorflow-gpu1.1

The training data and model is in the hyperlink below.  

https://pan.baidu.com/s/1KY_NOifssbJzutGgy96Fvw

https://pan.baidu.com/s/1AQjXIvw3JGZ1hMtyiQ7d7Q

https://pan.baidu.com/s/1-iV8gL7RhE8azniOx2G59g

https://pan.baidu.com/s/1bP0CV3nX4pDAZi0Yrq9tjg

https://pan.baidu.com/s/1NbHHWpFoJY5dFVv9St9ayw

And there's google link in case that you can not download from baidu:

https://drive.google.com/file/d/16cb1jGyXz4jcWRt0woGL85QyNkxGOQYT/view?usp=sharing

https://drive.google.com/drive/folders/1G3eeNw02bg3HajX-X8lKJEmkp_nk0IR9?usp=sharing

https://drive.google.com/drive/folders/1vBo9H7Uii5UJT7nTqX03uOA1EWPqdwp8?usp=sharing

https://drive.google.com/drive/folders/1QxQ1n0TgZYiFkMOI0Kqh3L3CAMXKNk3z?usp=sharing

https://drive.google.com/file/d/1JI24fp6sNEQqwvZhLip6eZbdpa9ThStc/view?usp=sharing

Before run the project make sure that you download all the files and models.

To start the test: just run the file test_one_sentence.py 

And the test_multi_sentence.py is for test multiple sentences once, please do not feed the files number over batchsize
test_short_sentence.py is for test the short sentence within 30 words with another model, because we find that the window length 250 is too long for the short sentence, which means that it will not have a good performance on short sentence.

If you want to train the model by yourself: 

python sentiment_analysis.py --isTrainingD2v='True' # to train the D2V part

python sentiment_analysis.py --isTrainingPredModel='True' # to train the sentiment analysis model

You can train the word2vec model by gensim, and later on i will upload the training file.

Don't worry, the trained word2vec model is in training data folder, which means that you can start the next part directly.
