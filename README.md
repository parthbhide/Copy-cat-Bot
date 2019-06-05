## Copy-cat-Bot
Generate plausible new text which looks like some other text !!

Idea taken from --->
https://github.com/parthbhide/awesome-project-ideas
Reference --->
mlm/blog

__________________________________________________________________________________________________________________________
#### Text Genration using LSTM 
#### Unsupervised Learning

__________________________________________________________________________________________________________________________

##### -->> Even with horrible accuracy of 26.47 percent can predict some meaningfull text, ( took 5.5 hours to train ), ( for better accuracy train with greater epochs on platforms like GCP )

__________________________________________________________________________________________________________________________

## Model Code :

#### #Modeldefination

#### #Takes parameter vocab_size and length of seed text

#### #Using one embedding layer, two LSTM layers and two dense layers

#### #Printing summary and returning model



def create_model(vocab_size,seq_ln):

    model = Sequential()
    
    model.add(Embedding(vocab_size,seq_ln,input_length=seq_ln))
    
    model.add(LSTM(80,return_sequences=True))
    
    model.add(LSTM(80))
    
    model.add(Dense(80,activation='relu'))
    
    model.add(Dense(vocab_size,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.summary()
    
    return model


____________________________________________________________________________________________________________

#### ------------------------| INPUT |-----------------------------

Seed text : --- > Let me begin by wishing ‘Namaskar’ to all of you

#### ------------------------| OUTPUT |-----------------------------

and thank you for this honour and you will have observed the starting effective role in the world of nuclear and agri vessels we have to do that in the world we have decided to do so as well as a great persona of the world we have a chance to be a very productive forum i am happy to be here to do so as well as a great proof of the world ’s largest and spiritual ranking in the world bank imf is a part of the world we have a new era of the world and the
