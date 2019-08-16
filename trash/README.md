## Trash classification
 >In this procedure, our model could identify 5 classes trash that the
  model would return some numbers like:0,  1, 2, 3, 4 and 5, if 5 is 
  your train model returned value, it is represent "I do not know this
   category trash."  
   
### Training step
* > Firstly, you must change some parameters in `datas.py`, including 
`csv_file_saved_path`, `self.image_path`.  
* >Secondly, may be you should know this procedure's process:
    1. **Renaming your training image:**
    In `datas.py` lines 59.
    2. **Resizing training image:**
    In `datas.py` lines 60.
    3. **Transferring your training image to csv file in order to our training model
    could read this training data conveniently.** In `datas.py` lines 61.
* > Then, Only run `datas.py` main function, you will get some files about 
train model, train logs and so on.
* >Finally, if you want to predict which category trash of the image, just remove
the `#` in the head of lines 66. this method need to parameters: the first is which 
data you want to recognize, and the second parameters is image's path. If the first 
parameter is `video`, you need not give the second parameter. But if the first parameter
is `image`, you have to give a correct image path that you want to classify as the second
parameter.

### About the project files

* `datas.py` is a program file that prepare the data, even include the main function of this project.
* `tf_model.py` is our definition of our model. And you maybe do not care this file.
* `train.py` is *Tensorflow* deeplearning framework's training step.


