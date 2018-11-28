# sentiment-analysis
A.Introduction and system environment
We build a simple system to automatically give labels to comments. 

We run the file under linux and win10, with python3.5.2 and tensorflow-gpu1.1.
If you want to run the system using web browser, make sure you add XAMPP in your pc.

B.Instruction to download data and model
The training data and model are available on Baidu in the hyperlinks below:
https://pan.baidu.com/s/1KY_NOifssbJzutGgy96Fvw

https://pan.baidu.com/s/1AQjXIvw3JGZ1hMtyiQ7d7Q

https://pan.baidu.com/s/1-iV8gL7RhE8azniOx2G59g

https://pan.baidu.com/s/1bP0CV3nX4pDAZi0Yrq9tjg

https://pan.baidu.com/s/1NbHHWpFoJY5dFVv9St9ayw

https://pan.baidu.com/s/1pGr5WnTpe-P3EwakhnsLzg

Alternatively, you might use below google links to download(HKU account only):
https://drive.google.com/file/d/16cb1jGyXz4jcWRt0woGL85QyNkxGOQYT/view?usp=sharing

https://drive.google.com/drive/folders/1G3eeNw02bg3HajX-X8lKJEmkp_nk0IR9?usp=sharing

https://drive.google.com/drive/folders/1vBo9H7Uii5UJT7nTqX03uOA1EWPqdwp8?usp=sharing

https://drive.google.com/drive/folders/1QxQ1n0TgZYiFkMOI0Kqh3L3CAMXKNk3z?usp=sharing

https://drive.google.com/file/d/1JI24fp6sNEQqwvZhLip6eZbdpa9ThStc/view?usp=sharing

https://drive.google.com/file/d/1dC5otbnXBksrFY9LpC5KaGCZuQA9pHHb/view?usp=sharing

Before you run the project, please make sure that you have downloaded all the files and models.

C.Instruction to start the testing:  

(i)Run the “test_one_sentence2.py” file for one sentence structure

(ii)Run the “test_multi_sentence.py” for multiple sentences. Please do not feed the files number over batchsize 

* “test_short_sentence.py” is to test the short sentence within 30 words with another model, because we find that the window length 250 is too long for the short sentence, which means that it will not have a good performance on short sentence.

D.Instruction to train the model by yourself:

python sentiment_analysis.py --isTrainingD2v='True' # to train the D2V part

python sentiment_analysis.py --isTrainingPredModel='True' # to train the sentiment analysis model

You can also train the word2vec model by gensim, and later on I will upload the training file.

In this project, the trained word2vec model is already in training data folder, which means that you can start the test already.

